import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS

PROMPT_V1_NAME = "trancao-thang-rag-prompt-v1"
PROMPT_V2_NAME = "trancao-thang-rag-prompt-v2"
SYSTEM_V1 = "You are a friendly RAG assistant. Answer in 2-4 concise sentences using only the supplied context. If unknown, say so.\n\nContext:\n{context}"
SYSTEM_V2 = "You are an expert information analyst. Identify relevant facts and give a clear, structured answer in 3-5 sentences. Do not infer beyond the context.\n\nContext:\n{context}"
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])

def push_prompts_to_hub(client):
    for name, prompt, desc in [(PROMPT_V1_NAME, PROMPT_V1, "V1 concise"), (PROMPT_V2_NAME, PROMPT_V2, "V2 structured")]:
        try:
            print(f"[push] {name} -> {client.push_prompt(name, object=prompt, description=desc)}")
        except Exception as exc:
            print(f"[push-error] {name}: {exc}")

def pull_prompts_from_hub(client):
    prompts = {}
    for name, local in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"[pull] {name} from Hub")
        except Exception as exc:
            raise RuntimeError(f"Prompt Hub pull failed for {name}: {exc}") from exc
    return prompts

def get_prompt_version(request_id):
    value = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if value % 2 == 0 else PROMPT_V2_NAME

@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question, version):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version, "context": context}

def setup_vectorstore():
    return build_vectorstore(split_text(load_knowledge_base(), 500, 50), get_embeddings())

def main():
    if not config.validate(): sys.exit(1)
    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)
    retriever = setup_vectorstore().as_retriever(search_kwargs={"k": 3})
    llm = get_llm(); counts = {"v1": 0, "v2": 0}
    for i, question in enumerate(SAMPLE_QUESTIONS):
        key = get_prompt_version(f"req-{i:04d}"); tag = "v1" if key == PROMPT_V1_NAME else "v2"
        ask_ab(retriever, llm, prompts[key], question, tag); counts[tag] += 1
        print(f"[{i+1:02d}] [prompt-{tag}] {question[:55]}...")
    print(f"Routing: V1={counts['v1']} | V2={counts['v2']} | total={len(SAMPLE_QUESTIONS)}")

if __name__ == "__main__": main()
