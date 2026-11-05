from __future__ import annotations
import pandas as pd
from modeling import recommend_products

DEFAULT_PRODUCTS = [
 {"name":"Recurring Deposit","category":"Savings","description":"A regular savings product for customers interested in disciplined deposits."},
 {"name":"Fixed Deposit","category":"Investment","description":"A fixed-term deposit product for customers exploring deposit investments."},
 {"name":"Mutual Fund Information","category":"Investment","description":"Educational information about mutual fund options; suitability requires separate assessment."},
 {"name":"Premium Debit Card","category":"Payments","description":"A payment card option for customers with frequent transaction activity."},
 {"name":"Digital Savings Tools","category":"Savings","description":"Digital tools for tracking balances and saving goals."},
 {"name":"Personal Finance Review","category":"Financial Wellness","description":"A review of spending, balances and financial goals."},
]

AGENT_NAMES = [
 "Data Ingestion Agent","Behaviour Analysis Agent","Pattern Detection Agent",
 "Life Event Detection Agent","Decision Agent","Product Recommendation Agent",
 "Message Generation Agent","Channel & Timing Agent","Fatigue Detection Agent",
 "Engagement Monitoring Agent"
]

def run_agent_pipeline(customer_id, customers, transactions, features, products=None, threshold=.5):
    products = products or DEFAULT_PRODUCTS
    customer = customers[customers["customer_id"].astype(str) == str(customer_id)].head(1)
    tx = transactions[transactions["customer_id"].astype(str) == str(customer_id)].copy()
    feature = features[features["customer_id"].astype(str) == str(customer_id)].head(1)
    if customer.empty or feature.empty:
        raise ValueError("Customer not found in loaded dataset.")
    row = feature.iloc[0]
    amount = float(row.get("total_transaction_amount",0) or 0)
    count = int(row.get("transaction_count",0) or 0)
    signals = []
    if amount > 0: signals.append("Recorded transaction activity")
    if float(row.get("total_investment",0) or 0) > 0: signals.append("Investment activity observed")
    if int(row.get("days_since_last_transaction",-1) or -1) >= 90: signals.append("No transaction in 90+ days")
    ranked = recommend_products(feature, products, threshold)
    selected = next((p for p in ranked if p["eligible"]), None)
    latest = tx["transaction_date"].max()
    agents = []
    def add(i, status, message, evidence=None):
        agents.append({"number":i+1,"name":AGENT_NAMES[i],"status":status,"message":message,"evidence":evidence or {}})
    add(0,"Completed","Customer and transaction records loaded.",{"customer_id":str(customer_id),"transactions":len(tx)})
    add(1,"Completed","Customer-level spending and investment features inspected.",{"transaction_count":count,"total_transaction_amount":amount,"total_investment":float(row.get("total_investment",0) or 0)})
    add(2,"Completed","Rule-based behavioural signals generated.",{"signals":signals or ["No strong signal"]})
    add(3,"Completed","Potential financial needs are inferred only as hypotheses; no verified life event is claimed.",{"possible_context":"Investment or savings interest" if row.get("total_investment",0)>0 else "General financial engagement"})
    decision = "Engagement considered" if selected else "No product passed relevance threshold"
    add(4,"Completed",decision,{"minimum_relevance":threshold,"candidate_count":len(ranked)})
    add(5,"Completed" if selected else "Suppressed","Product catalogue ranked using transparent heuristic relevance scores.",{"top_candidates":[{"product":p["name"],"score":round(p["score"],3)} for p in ranked[:3]]})
    message = (f"Based on your recorded banking activity, you may wish to explore {selected['name']}. "
               "This is an informational suggestion; review product terms and suitability before making a decision.") if selected else "No recommendation was generated because no product met the configured relevance threshold."
    add(6,"Completed" if selected else "Skipped","Personalised message template prepared." if selected else "No message generated without a qualifying recommendation.",{})
    channel = "In-app notification (simulated)"
    add(7,"Completed" if selected else "Skipped","A simulated channel and timing decision was made; no message is actually sent.",{"channel":channel if selected else "None","timing":"Next eligible engagement window"})
    add(8,"Completed","Notification fatigue rule checked. This prototype has no historical send log, so real frequency suppression is not measurable.",{"send_history_available":False})
    add(9,"Completed","Pipeline outcome recorded for dashboard display. No actual customer interaction is sent or tracked.",{"last_transaction_date":str(latest) if latest is not pd.NaT else "Unavailable"})
    return {"status":"Recommendation generated" if selected else "No recommendation","agents":agents,
            "recommendation":{"product":selected["name"],"score":selected["score"]} if selected else None,
            "message":message}
