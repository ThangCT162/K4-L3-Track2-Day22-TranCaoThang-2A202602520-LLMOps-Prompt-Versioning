# Evidence checklist

This directory contains the artifacts used to verify the four lab tasks.

| File | Verification |
|---|---|
| `01_langsmith_traces.png` | LangSmith project showing at least 50 `rag-query` traces and a trace with question, retrieved context, and answer. |
| `02_prompt_hub.png` | LangSmith Prompt Hub showing both personal prompt versions. |
| `02_ab_routing_log.txt` | 50 A/B requests with deterministic `prompt-v1`/`prompt-v2` labels. |
| `03_ragas_scores.png` | Comparison of the four RAGAS metrics for V1 and V2. |
| `03_ragas_report.json` | Machine-readable V1/V2 scores and `target_met`. |
| `04_pii_demo_log.txt` | PII test cases showing redacted output. |
| `04_json_demo_log.txt` | JSON repair test cases showing valid output. |

The LangSmith project should contain at least 100 traces in total: 50 from Task 1 and 50 from Task 2. Screenshots and logs must be generated from this repository's actual runs.
