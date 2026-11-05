from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

FEATURES = ["transaction_count","total_transaction_amount","average_transaction_amount","total_investment","average_balance","days_since_last_transaction"]

def _matrix(features):
    x = features[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).astype(float)
    return x

def customer_segments(features, max_clusters=4):
    if features.empty: return pd.DataFrame(columns=["customer_id","segment"])
    x = _matrix(features)
    n = len(x)
    if n < 3:
        labels = np.zeros(n, dtype=int)
    else:
        k = min(max_clusters, max(2, int(np.sqrt(n/2))), n)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(x))
    result = features[["customer_id"]].copy()
    result["segment"] = ["Segment " + str(int(v)+1) for v in labels]
    return result

def anomaly_scores(transactions):
    if transactions.empty:
        return pd.DataFrame(columns=["customer_id","anomaly_score","anomaly_flag"])
    t = transactions.copy()
    t["transaction_amount"] = pd.to_numeric(t["transaction_amount"],errors="coerce").fillna(0)
    t["investment_amount"] = pd.to_numeric(t["investment_amount"],errors="coerce").fillna(0)
    agg = t.groupby("customer_id").agg(
        tx_count=("transaction_amount","size"),
        avg_amount=("transaction_amount","mean"),
        max_amount=("transaction_amount","max"),
        investment=("investment_amount","sum")
    ).reset_index()
    if len(agg) < 5:
        agg["anomaly_score"] = 0.0
        agg["anomaly_flag"] = 0
        return agg[["customer_id","anomaly_score","anomaly_flag"]]
    model = IsolationForest(contamination="auto", random_state=42)
    x = agg[["tx_count","avg_amount","max_amount","investment"]].replace([np.inf,-np.inf],np.nan).fillna(0)
    model.fit(x)
    agg["anomaly_score"] = model.decision_function(x)
    agg["anomaly_flag"] = (model.predict(x) == -1).astype(int)
    return agg[["customer_id","anomaly_score","anomaly_flag"]]

def recommend_products(customer_row, products, threshold=.5):
    """Transparent heuristic recommender; score is a relevance heuristic, not a probability."""
    row = customer_row.iloc[0] if isinstance(customer_row, pd.DataFrame) else customer_row
    preferred = str(row.get("preferred_investment_type","")).lower()
    invest = float(row.get("total_investment",0) or 0)
    balance = float(row.get("average_balance",0) or 0)
    spend = float(row.get("total_transaction_amount",0) or 0)
    ranked = []
    for p in products:
        score = 0.35
        text = (p["category"]+" "+p["name"]+" "+p["description"]).lower()
        if preferred and preferred not in ["nan","unknown"] and any(k in text for k in preferred.split()):
            score += .25
        if invest > 0 and any(k in text for k in ["investment","mutual","deposit"]): score += .15
        if balance > 0 and any(k in text for k in ["savings","deposit","balance"]): score += .10
        if spend > 0 and any(k in text for k in ["payment","card","transaction"]): score += .05
        score = min(score, .95)
        ranked.append({**p,"score":score,"eligible":score >= threshold})
    ranked.sort(key=lambda x:x["score"], reverse=True)
    return ranked
