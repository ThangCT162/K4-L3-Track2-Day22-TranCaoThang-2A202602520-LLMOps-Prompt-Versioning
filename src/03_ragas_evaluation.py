import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

# Groq-compatible APIs only support one completion per request. RAGAS' default
# answer-relevancy strictness is 3, which sends n=3 and causes a 400 error.
answer_relevancy.strictness = 1
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS

SYSTEM_V1 = "Answer in 2-4 concise sentences using only the context. If unknown, say so.\n\nContext:\n{context}"
SYSTEM_V2 = "Give a clear structured answer in 3-5 sentences using only relevant facts from the context. Do not infer.\n\nContext:\n{context}"
PROMPTS = {"v1": ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")]), "v2": ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])}

def setup_vectorstore():
    return build_vectorstore(split_text(load_knowledge_base(), 500, 50), get_embeddings())

def run_rag(retriever, llm, prompt, question):
    contexts = [d.page_content for d in retriever.invoke(question)]
    answer = (prompt | llm | StrOutputParser()).invoke({"context": "\n\n".join(contexts), "question": question})
    return {"answer": answer, "contexts": contexts}

def collect_rag_outputs(vectorstore, version):
    checkpoint = Path(__file__).parent.parent / "data" / f"rag_{version}_checkpoint.json"
    results = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else []
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}); llm = get_llm()
    print(f"{version}: resume từ {len(results)}/{len(QA_PAIRS)} câu")
    for i, qa in enumerate(QA_PAIRS[len(results):], len(results) + 1):
        try:
            out = run_rag(retriever, llm, PROMPTS[version], qa["question"])
        except Exception as exc:
            msg = str(exc).lower()
            if any(x in msg for x in ("429", "rate limit", "quota", "resource_exhausted", "timeout", "503", "unavailable", "high demand")):
                checkpoint.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"\nSTOP: provider quota/timeout at {version} {i-1}/{len(QA_PAIRS)}.")
                print("Đổi API/model rồi chạy lại; checkpoint sẽ được tiếp tục.")
                raise SystemExit(2)
            raise
        results.append({"question": qa["question"], "reference": qa["reference"], "answer": out["answer"], "contexts": out["contexts"]})
        checkpoint.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i:02d}/50] {qa['question'][:60]} | checkpoint saved")
    return results

def build_ragas_dataset(rag_results):
    return EvaluationDataset(samples=[SingleTurnSample(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"], reference=r["reference"]) for r in rag_results])

def run_ragas_eval(rag_results, version):
    result = evaluate(
        build_ragas_dataset(rag_results),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=get_llm(temperature=0, max_tokens=4096),
        embeddings=get_embeddings(),
        run_config=RunConfig(max_workers=2, max_retries=3, max_wait=30, timeout=180),
        batch_size=5,
        raise_exceptions=False,
    )
    scores = {k: float(np.mean([v for v in result[k] if v is not None])) for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]}
    print(version, scores); return scores

def main():
    if not config.validate(): sys.exit(1)
    vs = setup_vectorstore(); v1 = run_ragas_eval(collect_rag_outputs(vs, "v1"), "v1"); v2 = run_ragas_eval(collect_rag_outputs(vs, "v2"), "v2")
    report = {"prompt_v1_scores": v1, "prompt_v2_scores": v2, "target_met": max(v1["faithfulness"], v2["faithfulness"]) >= 0.8}
    path = Path(__file__).parent.parent / "data" / "ragas_report.json"; path.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(path)

if __name__ == "__main__": main()
