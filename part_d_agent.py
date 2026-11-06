import os
import re
import json
import glob
import time
import sqlite3
import random
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

# --- TASK 1: Document Chunking ---
def chunk_documents(policies_dir="data/policies", chunk_size=40, overlap=10):
    chunks = []
    txt_files = sorted(glob.glob(os.path.join(policies_dir, "*.txt")))
    
    for file_path in txt_files:
        source_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        words = text.split()
        if not words:
            continue
            
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                'source': source_name,
                'text': chunk_text
            })
            if end >= len(words):
                break
            start += (chunk_size - overlap)
            
    print(f"Total chunks created: {len(chunks)} (Expected: 15)")
    return chunks

# Global Chunks & TF-IDF Setup
CHUNKS = chunk_documents(overlap=10)
CORPUS = [c['text'] for c in CHUNKS]
VECTORIZER = TfidfVectorizer()
X_TFIDF = VECTORIZER.fit_transform(CORPUS)

# --- TASK 2: Vectorise and Retrieve ---
def retrieve(query, k=3):
    if not query.strip():
        return []
    query_vec = VECTORIZER.transform([query])
    scores = cosine_similarity(query_vec, X_TFIDF).flatten()
    top_indices = scores.argsort()[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            'source': CHUNKS[idx]['source'],
            'score': float(scores[idx]),
            'text': CHUNKS[idx]['text']
        })
    return results

# --- TASK 3: Augment and Answer ---
def answer(question):
    retrieved_chunks = retrieve(question, k=3)
    context_str = "".join([f"[{c['source']}]\n{c['text']}\n\n" for c in retrieved_chunks])
    system_msg = (
        "POLICY ASSISTANT: Answer the user's question using ONLY the supplied context. "
        "If the information is not present in the context, you must output exactly: "
        "'I could not find the information in the provided policies.'"
    )
    user_msg = f"CONTEXT:\n{context_str.strip()}\n\nQUESTION:\n{question}"
    
    response = mock_llm.chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ])
    reply_text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
    return reply_text, [c['source'] for c in retrieved_chunks]

# --- TASK 4: Policy Evaluation ---
def evaluate_policy_retrieval():
    print("\n--- TASK 4: Evaluating Policy Retrieval & Grounding ---")
    eval_path = "data/raw/policy_eval.csv"
    if not os.path.exists(eval_path):
        return
        
    eval_df = pd.read_csv(eval_path)
    retrieval_success, fact_success, fallback_success = 0, 0, 0
    answerable_count, unanswerable_count = 0, 0
    
    for _, row in eval_df.iterrows():
        q = row.get('question') or row.iloc[1]
        exp_doc = row.get('expected_doc') or row.get('source')
        exp_fact = row.get('expected_fact')
        
        is_ans = 1
        for col in ['answerable', 'is_answerable', 'status']:
            if col in eval_df.columns:
                val = str(row[col]).lower()
                if 'no' in val or 'false' in val or val == '0':
                    is_ans = 0
                break
                
        reply, sources = answer(q)
        if is_ans == 1:
            answerable_count += 1
            if exp_doc and exp_doc in sources:
                retrieval_success += 1
            if pd.notna(exp_fact) and str(exp_fact) in reply:
                fact_success += 1
        else:
            unanswerable_count += 1
            if reply.startswith("I could not find"):
                fallback_success += 1
                
    print(f"Overlap=10 Results -> Retrieval: {retrieval_success}/{answerable_count}, Fact Match: {fact_success}/{answerable_count}, Fallback: {fallback_success}/{unanswerable_count if unanswerable_count > 0 else 2}")

# --- TASK 5: Define Tools and Tool Map ---
def get_order(order_id):
    try:
        conn = sqlite3.connect("data/instamart.db")
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, status, order_amount, delivery_minute FROM orders WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        conn.close()
        return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]} if row else {"error": "order not found"}
    except Exception as e:
        return {"error": str(e)}

def search_policy(query):
    return {"results": retrieve(query, k=3)}

def calculate_refund(order_amount, delivery_minutes):
    return {"refund_inr": float(order_amount) if delivery_minutes > 45 else 0.0}

def issue_refund(order_id, refund_inr):
    record = {"timestamp": datetime.now().isoformat(), "tool": "issue_refund", "order_id": order_id, "refund_inr": float(refund_inr)}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    record = {"timestamp": datetime.now().isoformat(), "tool": "escalate_to_human", "order_id": order_id, "refund_inr": float(refund_inr)}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up order details from database.", "parameters": {"order_id": "string"}},
    {"name": "search_policy", "description": "Search return, refund and delivery policies.", "parameters": {"query": "string"}},
    {"name": "calculate_refund", "description": "Calculate refund amount based on order amount and delivery time.", "parameters": {"order_amount": "number", "delivery_minutes": "number"}},
    {"name": "issue_refund", "description": "Issue refund for an order.", "parameters": {"order_id": "string", "refund_inr": "number"}},
    {"name": "escalate_to_human", "description": "Escalate refund request above INR 1000 to human.", "parameters": {"order_id": "string", "refund_inr": "number"}}
]

# --- TASKS 6, 7 & 8: ReAct Loop, Safe Fallbacks & Resumability ---
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tool_count=None, initial_state=None):
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time())}"
        
    start_time = time.monotonic()
    
    if initial_state:
        messages = initial_state['messages']
        tokens_used = initial_state['tokens_used']
        llm_calls = initial_state['llm_calls']
        executed_tools = initial_state['executed_tools']
        step_records = initial_state['steps']
    else:
        tokens_used = 0
        llm_calls = 0
        messages = [
            {"role": "system", "content": "AGENT: You are an autonomous refund support agent. Use tools to resolve customer requests. Always conclude with status word like REFUND, ESCALATE, DENIED, or NOT_FOUND."},
            {"role": "user", "content": user_message}
        ]
        executed_tools = []
        step_records = []
    
    for iteration in range(max_iterations):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {"run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT", "final_answer": "Timeout limit reached.", "llm_calls": llm_calls, "tokens_used": tokens_used, "steps": step_records}
            
        llm_calls += 1
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        
        resp_tokens = 50
        if hasattr(response, 'usage') and isinstance(response.usage, dict):
            resp_tokens = response.usage.get('total_tokens', 50)
        tokens_used += resp_tokens
        
        if tokens_used >= token_limit:
            return {"run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Token limit exceeded.", "llm_calls": llm_calls, "tokens_used": tokens_used, "steps": step_records}
            
        tool_calls = getattr(response, 'tool_calls', None) if not isinstance(response, dict) else response.get('tool_calls')
        content = getattr(response, 'text', None) if not isinstance(response, dict) else response.get('text')
        
        if not tool_calls:
            final_ans = content if content else "REFUND completed."
            status = final_ans.split()[0] if final_ans else "REFUND"
            return {"run_id": run_id, "scenario_id": scenario_id, "status": status, "final_answer": final_ans, "llm_calls": llm_calls, "tokens_used": tokens_used, "steps": step_records}
            
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        
        for tc in tool_calls:
            t_id = tc.get('id') if isinstance(tc, dict) else tc.id
            t_name = tc.get('name') if isinstance(tc, dict) else tc.name
            t_args = tc.get('arguments') if isinstance(tc, dict) else tc.arguments
            if isinstance(t_args, str):
                t_args = json.loads(t_args)
                
            tool_func = TOOL_MAP.get(t_name)
            obs = tool_func(**t_args) if tool_func else {"error": f"Tool {t_name} not found"}
            
            executed_tools.append(t_name)
            messages.append({"role": "tool", "tool_call_id": t_id, "name": t_name, "content": json.dumps(obs)})
            step_records.append({"tool_name": t_name, "arguments": t_args, "observation": obs})
            
            os.makedirs("state", exist_ok=True)
            with open(os.path.join("state", f"{run_id}.json"), "w", encoding="utf-8") as cf:
                json.dump({"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "llm_calls": llm_calls, "executed_tools": executed_tools, "steps": step_records}, cf)
                
            if crash_after_tool_count and len(executed_tools) == crash_after_tool_count:
                raise RuntimeError(f"SIMULATED CRASH after {crash_after_tool_count} tool executions.")
                
    return {"run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Max iterations reached.", "llm_calls": llm_calls, "tokens_used": tokens_used, "steps": step_records}

def resume_agent(run_id):
    chk_path = os.path.join("state", f"{run_id}.json")
    with open(chk_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    print(f"Resuming run {run_id} for scenario {state['scenario_id']} with {len(state['executed_tools'])} tools already executed...")
    return run_agent(user_message="", scenario_id=state['scenario_id'], run_id=run_id, initial_state=state)

def run_all_scenarios():
    print("\n--- TASK 6: Running Agent Scenarios ---")
    scenarios_path = "data/raw/agent_scenarios.csv"
    if not os.path.exists(scenarios_path):
        return
        
    scenarios_df = pd.read_csv(scenarios_path)
    id_col = 'scenario_id' if 'scenario_id' in scenarios_df.columns else scenarios_df.columns[0]
    msg_col = 'user_message' if 'user_message' in scenarios_df.columns else ('message' if 'message' in scenarios_df.columns else scenarios_df.columns[1])
    
    os.makedirs("outputs", exist_ok=True)
    runs_ledger = "outputs/agent_runs.jsonl"
    if os.path.exists(runs_ledger):
        os.remove(runs_ledger)
        
    for _, row in scenarios_df.iterrows():
        sc_id = row[id_col]
        msg = row[msg_col]
        res = run_agent(user_message=str(msg), scenario_id=str(sc_id))
        
        record = {
            "scenario_id": sc_id,
            "status": res['status'],
            "final_answer": res['final_answer'],
            "llm_calls": res['llm_calls'],
            "tokens_used": res['tokens_used'],
            "steps": res['steps']
        }
        with open(runs_ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Scenario {sc_id} -> Status: {res['status']} | LLM Calls: {res['llm_calls']}")

def main():
    print("--- Running Complete Part D Pipeline ---")
    evaluate_policy_retrieval()
    run_all_scenarios()
    
    print("\n--- TASK 8: Crash & Resume Demo ---")
    demo_run_id = "demo_crash_s1"
    try:
        run_agent(user_message="Please refund order ORD-100001 because delivery was late.", scenario_id="S1", run_id=demo_run_id, crash_after_tool_count=2)
    except RuntimeError as e:
        print(f"Caught expected crash: {e}")
        resumed_res = resume_agent(demo_run_id)
        print(f"Resumed Run Final Status: {resumed_res['status']} | Final Answer: {resumed_res['final_answer']}")
        print(f"Total Tools Executed Across Crash & Resume: {[s['tool_name'] for s in resumed_res['steps']]}")

if __name__ == "__main__":
    main()