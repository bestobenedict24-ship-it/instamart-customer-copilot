# Instamart Customer Intelligence & Support Copilot

An end-to-end AI-powered customer intelligence and automated support copilot built for quick-commerce operations. This project integrates customer analytics, churn prediction, automated ticket triage with PII redaction, policy RAG retrieval, and an autonomous ReAct refund agent with safety fallbacks and crash-resumption checkpointing.

---

## Overview
This project provides a comprehensive quick-commerce solution covering customer data analysis, support ticket safety, policy Q&A, and an autonomous refund execution agent.

## Reproduce
1. Activate your virtual environment: `venv\Scripts\activate`
2. Install dependencies: `pip install pandas numpy scikit-learn`
3. Run the complete Part D pipeline: `python part_d_agent.py`

---

## Complete Project Breakdown

### Part A: Exploratory Data Analysis & Schema Verification
- **SQLite Database Integration**: Queried and validated core tables (`customers`, `orders`, `support_tickets`, `refunds`) in `data/instamart.db`.
- **Data Integrity & Metrics**: Calculated baseline order volume, average order value (AOV), delivery latency distributions, and initial ticket issue distributions.

### Part B: Customer Segmentation & Lapse Prediction
- **Feature Engineering**: Built RFM (Recency, Frequency, Monetary) metrics and operational parameters (late delivery count, refund history).
- **Segmentation Model**: Categorized customers into key operational buckets (e.g., High-Value Loyal, At-Risk, Lapsed).
- **Lapse Risk Classifier**: Trained and evaluated predictive models to identify customers likely to churn due to order delays or support friction.

### Part C: Ticket Triage, PII Redaction & Guardrails
- **Automated Triage Engine**: Categorized customer support tickets by issue type and urgency tier.
- **PII Masking & Safety**: Implemented regex-based redaction for phone numbers, email addresses, and credit card numbers prior to LLM processing.
- **Safety Guardrails**: Implemented input/output filtering to prevent toxic prompts and out-of-scope customer queries.

### Part D: Policy RAG Assistant & Autonomous Refund Agent
- **Task 1 & 2 (Chunking & Retrieval)**: Chunked policy text documents (`data/policies/`) into 40-word segments with 10-word overlap (15 total chunks) and built a TF-IDF cosine-similarity retrieval engine.
- **Task 3 & 4 (RAG QA & Policy Evaluation)**: Developed grounded policy QA (`answer()`) using `mock_llm.chat` and evaluated retrieval accuracy over `policy_eval.csv` achieving **12/14 retrieval and fact-match accuracy**.
- **Task 5 (Agent Tool Suite)**: Defined functional tool wrappers (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`).
- **Task 6 & 7 (ReAct Agent & Fallbacks)**: Constructed autonomous loop handling customer scenarios (`agent_scenarios.csv`) across 7 scenarios with strict guardrails (max iterations, token caps, time-limits).
- **Task 8 (State Checkpointing & Crash Resumption)**: Implemented mid-turn state persistence to `state/*.json` allowing interrupted agent runs to resume without repeating executed tools.

---

## Limitations and Next Steps
- **Limitations**: TF-IDF retrieval works well for keyword-heavy policies but can struggle with semantic paraphrasing compared to dense vector embeddings.
- **Next Steps**: Integrate vector embeddings (e.g., OpenAI or Hugging Face embeddings) with a vector database for semantic search, and deploy the agent via a web interface (Streamlit).

## Project Structure & Architecture

```text
instamart-customer-copilot/
│
├── data/
│   ├── policies/          # Policy text documents
│   ├── instamart.db       # SQLite database containing order and customer records
│   └── raw/               # Evaluation datasets
│
├── outputs/               # Generated logs and ledgers
│   ├── agent_runs.jsonl
│   └── refund_ledger.jsonl
│
├── sql/
│   └── queries.sql        # SQL analytics & schema verification queries for Part A
│
├── state/                 # JSON execution checkpoints
├── .gitignore             # Git ignore configuration
├── requirements.txt       # Project dependencies (pandas, numpy, scikit-learn)
├── part_d_agent.py        # Core Part D execution pipeline
└── README.md              # Project documentation