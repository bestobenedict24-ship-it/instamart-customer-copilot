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

### Part A: SQL Analytics
- **Code Snippet (`sql/queries.sql`)**:
  ```sql
  -- Query 1: Total Volume & Order Summary Metrics
  SELECT 
      COUNT(order_id) AS total_orders,
      COUNT(DISTINCT customer_id) AS total_unique_customers,
      ROUND(AVG(order_amount), 2) AS average_order_value,
      SUM(order_amount) AS total_revenue
  FROM orders;


### Part B: Customer Segmentation & Lapse Prediction
- **Feature Engineering**: Built RFM (Recency, Frequency, Monetary) metrics and operational parameters (late delivery count, refund history).
- **Segmentation Model**: Categorized customers into key operational buckets (e.g., High-Value Loyal, At-Risk, Lapsed).
- **Lapse Risk Classifier**: Trained and evaluated predictive models to identify customers likely to churn due to order delays or support friction.

## Core Implementation & Code Highlights

### Part B: Customer Segmentation & Churn Modeling
* **RFM Feature Engineering & Lapse Risk Classifier (`part_b_rfm_clusters.py` & `part_b_lapse_model.py`)**:
  ```python
  import pandas as pd
  from sklearn.ensemble import RandomForestClassifier

  # Aggregate Recency, Frequency, and Monetary metrics
  rfm = df.groupby('customer_id').agg({
      'order_date': lambda x: (snapshot_date - x.max()).days,
      'order_id': 'count',
      'order_amount': 'sum'
  }).rename(columns={'order_date': 'Recency', 'order_id': 'Frequency', 'order_amount': 'Monetary'})

  # Train lapse risk model based on delivery friction & support complaints
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

### Part C: Ticket Triage, PII Redaction & Guardrails
- **Automated Triage Engine**: Categorized customer support tickets by issue type and urgency tier.
- **PII Masking & Safety**: Implemented regex-based redaction for phone numbers, email addresses, and credit card numbers prior to LLM processing.
- **Safety Guardrails**: Implemented input/output filtering to prevent toxic prompts and out-of-scope customer queries.

### Part C: Ticket Triage, PII Redaction & Guardrails
- **Code Snippet (`part_c_triage.py`)**:
  ```python
  import re

  def redact_pii(text: str) -> str:
      """Redact sensitive customer details (Email, Phone, Credit Card) prior to LLM processing."""
      text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', text)
      text = re.sub(r'\b\d{10}\b', '[PHONE_MASKED]', text)
      text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CARD_MASKED]', text)
      return text

  def triage_ticket(ticket_text: str):
      clean_text = redact_pii(ticket_text)
      # Route to category & urgency tier
      return {"sanitized_text": clean_text, "urgency": "HIGH", "category": "Delivery Delay"}

### Part D: Policy RAG Assistant & Autonomous Refund Agent
- **Task 1 & 2 (Chunking & Retrieval)**: Chunked policy text documents (`data/policies/`) into 40-word segments with 10-word overlap (15 total chunks) and built a TF-IDF cosine-similarity retrieval engine.
- **Task 3 & 4 (RAG QA & Policy Evaluation)**: Developed grounded policy QA (`answer()`) using `mock_llm.chat` and evaluated retrieval accuracy over `policy_eval.csv` achieving **12/14 retrieval and fact-match accuracy**.
- **Task 5 (Agent Tool Suite)**: Defined functional tool wrappers (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`).
- **Task 6 & 7 (ReAct Agent & Fallbacks)**: Constructed autonomous loop handling customer scenarios (`agent_scenarios.csv`) across 7 scenarios with strict guardrails (max iterations, token caps, time-limits).
- **Task 8 (State Checkpointing & Crash Resumption)**: Implemented mid-turn state persistence to `state/*.json` allowing interrupted agent runs to resume without repeating executed tools.

### Part D: Policy RAG Assistant & Autonomous Refund Agent
- **Code Snippet (`part_d_agent.py`)**:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  # Step 1: Policy Retrieval Engine using TF-IDF
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(policy_chunks)

  def search_policy(query: str, top_k: int = 2):
      query_vec = vectorizer.transform([query])
      scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
      top_indices = scores.argsort()[-top_k:][::-1]
      return [policy_chunks[i] for i in top_indices]

  # Step 2: Autonomous ReAct Tool Wrapper Execution
  def execute_tool(tool_name: str, **kwargs):
      if tool_name == "calculate_refund":
          return calculate_refund_amount(kwargs.get("order_id"))
      elif tool_name == "issue_refund":
          return issue_refund_transaction(kwargs.get("order_id"), kwargs.get("amount"))
      elif tool_name == "escalate_to_human":
          return escalate_ticket(kwargs.get("reason"))

---

## Limitations and Next Steps
- **Limitations**: TF-IDF retrieval works well for keyword-heavy policies but can struggle with semantic paraphrasing compared to dense vector embeddings.
- **Next Steps**: Integrate vector embeddings (e.g., OpenAI or Hugging Face embeddings) with a vector database for semantic search, and deploy the agent via a web interface (Streamlit).

## Project Structure & Architecture

```text
📁 instamart-customer-copilot
├── 📁 data
│   ├── 📁 policies
│   ├── 📄 instamart.db
│   └── 📁 raw
├── 📁 outputs
│   ├── 📄 agent_runs.jsonl
│   └── 📄 refund_ledger.jsonl
├── 📁 sql
│   └── 📄 queries.sql
├── 📁 state
├── 📄 .gitignore
├── 📄 audit_data.py
├── 📄 build_profiles.py
├── 📄 clean_data.py
├── 📄 generate_data.py
├── 📄 mock_llm.py
├── 📄 part_b_final.py
├── 📄 part_b_lapse_model.py
├── 📄 part_b_rfm_clusters.py
├── 📄 part_c_triage.py
├── 📄 part_d_agent.py
├── 📄 README.md
└── 📄 requirements.txt