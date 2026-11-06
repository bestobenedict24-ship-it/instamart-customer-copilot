import os
import re
import random
import time
import pandas as pd
import numpy as np
import mock_llm

# --- Configuration & Constants ---
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

# Regex patterns for Task 2
EMAIL_REGEX = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
PHONE_REGEX = re.compile(r'\b[6-9]\d{9}\b')
ORDER_REGEX = re.compile(r'ORD-\d{6}')

ALLOWED_CATEGORIES = ['billing', 'delivery', 'technical', 'fraud']

# --- Prompts for Task 3 ---
ZERO_SHOT_SYSTEM = (
    "CLASSIFY the given support ticket into one of these four categories: billing, delivery, technical, fraud. "
    "Return a JSON-only reply of the exact form: {\"category\": \"...\", \"reason\": \"...\"}."
)

FEW_SHOT_SYSTEM = (
    "CLASSIFY the given support ticket into one of these four categories: billing, delivery, technical, fraud. "
    "Return a JSON-only reply of the exact form: {\"category\": \"...\", \"reason\": \"...\"}.\n\n"
    "EXAMPLES:\n"
    "Ticket: 'My account was charged twice for order ORD-123456.'\n"
    "Response: {\"category\": \"billing\", \"reason\": \"Duplicate charge reported on order.\"}\n\n"
    "Ticket: 'Someone used my credit card to place an unauthorized order ORD-999888!'\n"
    "Response: {\"category\": \"fraud\", \"reason\": \"Unauthorized transaction reported by customer.\"}"
)

def est_tokens(text):
    """Estimate token count."""
    if not isinstance(text, str):
        return 0
    return len(text.split())

def redact_text(text):
    """Redacts email, phone, and order IDs according to strict regex rules."""
    if not isinstance(text, str):
        return text, 0, 0, 0
    
    text, email_count = EMAIL_REGEX.subn('[EMAIL]', text)
    text, phone_count = PHONE_REGEX.subn('[PHONE]', text)
    text, order_count = ORDER_REGEX.subn('[ORDER_ID]', text)
    
    total_subs = email_count + phone_count + order_count
    return text, email_count, phone_count, order_count

def call_llm_with_retry(system_prompt, user_content, ticket_id):
    """Wraps model call with max 3 retries, exponential backoff, and jitter using mock_llm.chat."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    for n in range(3):
        try:
            response = mock_llm.chat(messages)
            return response, None
        except mock_llm.TransientLLMError as e:
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"[RETRY] Ticket {ticket_id} | Retry #{n} | Wait: {wait:.4f}s | Error: {e}")
            time.sleep(wait)
        except Exception as e:
            return None, e
    return None, "Max retries exceeded with TransientLLMErrors"

def run_pipeline(prompt_type="zero_shot", token_budget=None):
    print(f"\n=== RUNNING PIPELINE: {prompt_type.upper()} (Budget: {token_budget}) ===")
    mock_llm.reset()
    
    tickets_df = pd.read_csv("data/raw/tickets.csv")
    system_prompt = ZERO_SHOT_SYSTEM if prompt_type == "zero_shot" else FEW_SHOT_SYSTEM
    
    results = []
    total_tokens_used = 0
    
    email_subs_total = 0
    phone_subs_total = 0
    order_subs_total = 0
    
    for idx, row in tickets_df.iterrows():
        ticket_id = row['ticket_id']
        customer_id = row['customer_id']
        raw_text = str(row['text'])
        true_category = row['true_category']
        
        # Task 1: Input Guardrails
        if not raw_text or raw_text.strip() == "" or raw_text.lower() == "nan":
            results.append({
                'ticket_id': ticket_id, 'customer_id': customer_id, 'status': 'skipped_empty',
                'category': None, 'reason': None, 'tokens_used': 0, 'true_category': true_category, 'raw_text': raw_text
            })
            continue
            
        if est_tokens(raw_text) > MAX_INPUT_TOKENS:
            results.append({
                'ticket_id': ticket_id, 'customer_id': customer_id, 'status': 'flagged_long',
                'category': None, 'reason': None, 'tokens_used': 0, 'true_category': true_category, 'raw_text': raw_text
            })
            continue
            
        # Task 2: Redaction
        redacted, e_c, p_c, o_c = redact_text(raw_text)
        email_subs_total += e_c
        phone_subs_total += p_c
        order_subs_total += o_c
        
        # Assertion: Ensure no PII left in redacted text sent to model
        assert not EMAIL_REGEX.search(redacted), f"Email found in redacted text for {ticket_id}"
        assert not PHONE_REGEX.search(redacted), f"Phone found in redacted text for {ticket_id}"
        assert not ORDER_REGEX.search(redacted), f"Order ID found in redacted text for {ticket_id}"
        
        # Task 7a: Token Budget check
        if token_budget is not None and total_tokens_used >= token_budget:
            results.append({
                'ticket_id': ticket_id, 'customer_id': customer_id, 'status': 'budget_exhausted',
                'category': None, 'reason': None, 'tokens_used': 0, 'true_category': true_category, 'raw_text': redacted
            })
            continue
            
        # Task 4: Resilient Model Call
        response, err = call_llm_with_retry(system_prompt, redacted, ticket_id)
        
        if err or response is None:
            results.append({
                'ticket_id': ticket_id, 'customer_id': customer_id, 'status': 'flagged_invalid_output',
                'category': None, 'reason': str(err), 'tokens_used': 0, 'true_category': true_category, 'raw_text': redacted
            })
            continue
            
        # Handle response text and token tracking safely
        if hasattr(response, 'text'):
            reply_text = response.text.strip()
            tokens = getattr(response, 'usage', {}).get('total_tokens', len(reply_text.split())) if hasattr(response, 'usage') else len(reply_text.split())
        elif isinstance(response, str):
            reply_text = response.strip()
            tokens = len(reply_text.split()) + len(redacted.split())
        else:
            reply_text = str(response).strip()
            tokens = 50
            
        total_tokens_used += tokens
        
        # Task 5: Validate Every Answer (JSON vs Keyword fallback)
        category = None
        status = None
        reason = None
        
        try:
            import json
            parsed = json.loads(reply_text)
            cat_candidate = parsed.get('category', '').lower().strip()
            if cat_candidate in ALLOWED_CATEGORIES:
                category = cat_candidate
                status = 'classified_json'
                reason = parsed.get('reason', '')
            else:
                status = 'flagged_invalid_output'
        except Exception:
            words = re.findall(r"[a-z]+", reply_text.lower())
            found_cats = [w for w in words if w in ALLOWED_CATEGORIES]
            if len(found_cats) == 1:
                category = found_cats[0]
                status = 'classified_keyword'
                reason = "Extracted via keyword fallback"
            else:
                status = 'flagged_invalid_output'
                
        results.append({
            'ticket_id': ticket_id, 'customer_id': customer_id, 'status': status,
            'category': category, 'reason': reason, 'tokens_used': tokens, 'true_category': true_category, 'raw_text': redacted
        })
        
    print(f"Total substitutions across tickets -> Emails: {email_subs_total}, Phones: {phone_subs_total}, Orders: {order_subs_total}")
    return pd.DataFrame(results), total_tokens_used

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # Run Zero-Shot and Few-Shot comparisons (Task 6)
    df_zero, tokens_zero = run_pipeline("zero_shot")
    df_few, tokens_few = run_pipeline("few_shot")
    
    print("\n--- Zero-Shot Status Counts ---")
    print(df_zero['status'].value_counts())
    
    classified_zero = df_zero[df_zero['status'].isin(['classified_json', 'classified_keyword'])]
    acc_zero = (classified_zero['category'] == classified_zero['true_category']).mean() * 100
    print(f"Zero-Shot Accuracy over {len(classified_zero)} classified tickets: {acc_zero:.1f}%")
    
    classified_few = df_few[df_few['status'].isin(['classified_json', 'classified_keyword'])]
    acc_few = (classified_few['category'] == classified_few['true_category']).mean() * 100
    print(f"Few-Shot Accuracy over {len(classified_few)} classified tickets: {acc_few:.1f}%")
    
    # Task 7a: Demo budget run (Zero-shot with Budget = 6000)
    df_demo, _ = run_pipeline("zero_shot", token_budget=6000)
    exhausted_count = (df_demo['status'] == 'budget_exhausted').sum()
    print(f"\nDemo Budget Run: {exhausted_count} tickets ended budget_exhausted.")
    df_demo[['ticket_id', 'customer_id', 'status', 'category', 'tokens_used']].to_csv("outputs/triage_budget_demo.csv", index=False)
    
    # Task 7b: Final Production Routing using Few-Shot and Segment Data
    segments_df = pd.read_csv("outputs/customer_segments.csv")
    premium_customers = set(segments_df[segments_df['is_premium'] == True]['customer_id'])
    
    final_rows = []
    for _, row in df_few.iterrows():
        status = row['status']
        cat = row['category']
        cust_id = row['customer_id']
        
        seg_row = segments_df[segments_df['customer_id'] == cust_id]
        seg_name = seg_row['segment_name'].values[0] if not seg_row.empty else ''
        is_prem = cust_id in premium_customers
        
        if status in ['skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted']:
            route = 'human_review'
            priority = 'NORMAL'
        elif cat == 'fraud':
            route = 'security_team'
            priority = 'HIGH'
        elif is_prem:
            route = 'premium_desk'
            priority = 'HIGH'
        else:
            route = 'support_queue'
            priority = 'NORMAL'
            
        final_rows.append({
            'ticket_id': row['ticket_id'],
            'customer_id': cust_id,
            'status': status,
            'category': cat if pd.notna(cat) else '',
            'segment_name': seg_name if pd.notna(seg_name) else '',
            'priority': priority,
            'route': route,
            'tokens_used': row['tokens_used']
        })
        
    final_results_df = pd.DataFrame(final_rows)
    final_results_df.to_csv("outputs/triage_results.csv", index=False)
    print(f"\nFinal triage results saved to outputs/triage_results.csv with {len(final_results_df)} rows.")

if __name__ == "__main__":
    main()