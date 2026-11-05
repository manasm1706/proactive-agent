from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

def _read(source, filename):
    if source is not None:
        return pd.read_csv(source)
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def load_dataset(customer_file=None, transaction_file=None, bank_file=None):
    customers = _read(customer_file, "customer_data.csv")
    transactions = _read(transaction_file, "transaction_data.csv")
    banks = _read(bank_file, "bank_data.csv")
    if customers.empty or transactions.empty:
        raise ValueError("Customer and transaction CSVs are required. Upload them in the sidebar or place them in data/.")
    return customers, transactions, banks

def _normalise_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df

def _rename_first(df, target, candidates):
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: target})
            return df
    return df

def prepare_data(customers, transactions, banks):
    customers = _normalise_columns(customers)
    transactions = _normalise_columns(transactions)
    banks = _normalise_columns(banks)
    customers = _rename_first(customers, "customer_id", ["customer_id","customerid","customer_key"])
    customers = _rename_first(customers, "age", ["age","customer_age"])
    customers = _rename_first(customers, "customer_type", ["customer_type","type"])
    customers = _rename_first(customers, "city", ["city","customer_city"])
    customers = _rename_first(customers, "region", ["region","state","customer_region"])
    transactions = _rename_first(transactions, "transaction_id", ["transaction_id","transactionid"])
    transactions = _rename_first(transactions, "customer_id", ["customer_id","customerid","customer_key"])
    transactions = _rename_first(transactions, "transaction_date", ["transaction_date","date","transaction_datetime"])
    transactions = _rename_first(transactions, "transaction_amount", ["transaction_amount","amount","transaction_value"])
    transactions = _rename_first(transactions, "total_balance", ["total_balance","balance","account_balance"])
    transactions = _rename_first(transactions, "investment_amount", ["investment_amount","invested_amount"])
    transactions = _rename_first(transactions, "investment_type", ["investment_type","investment_category"])
    transactions = _rename_first(transactions, "account_type", ["account_type","account"])
    for c in ["transaction_amount","total_balance","investment_amount"]:
        if c not in transactions: transactions[c] = np.nan
        transactions[c] = pd.to_numeric(transactions[c], errors="coerce")
    if "transaction_date" not in transactions: transactions["transaction_date"] = pd.NaT
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], errors="coerce", dayfirst=False)
    if "customer_id" not in customers or "customer_id" not in transactions:
        raise ValueError("Both customer_data.csv and transaction_data.csv must contain Customer_ID.")
    customers["customer_id"] = customers["customer_id"].astype(str)
    transactions["customer_id"] = transactions["customer_id"].astype(str)
    if "city" not in customers: customers["city"] = "Unknown"
    if "region" not in customers: customers["region"] = "Unknown"
    if "customer_type" not in customers: customers["customer_type"] = "Unknown"
    if "age" in customers: customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    if banks is not None and not banks.empty:
        banks = _rename_first(banks, "branch_id", ["branch_id","branchid"])
        for c in ["firm_revenue","expenses","profit_margin"]:
            if c in banks: banks[c] = pd.to_numeric(banks[c], errors="coerce")
    return customers, transactions, banks

def make_customer_features(customers, transactions):
    tx = transactions.copy()
    tx["transaction_amount"] = tx["transaction_amount"].fillna(0)
    tx["investment_amount"] = tx["investment_amount"].fillna(0)
    grouped = tx.groupby("customer_id").agg(
        transaction_count=("transaction_id","count") if "transaction_id" in tx else ("customer_id","size"),
        total_transaction_amount=("transaction_amount","sum"),
        average_transaction_amount=("transaction_amount","mean"),
        total_investment=("investment_amount","sum"),
        average_balance=("total_balance","mean"),
        latest_transaction=("transaction_date","max"),
        first_transaction=("transaction_date","min"),
    ).reset_index()
    if "investment_type" in tx:
        mode = tx.groupby("customer_id")["investment_type"].agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else "Unknown").rename("preferred_investment_type").reset_index()
        grouped = grouped.merge(mode, on="customer_id", how="left")
    else: grouped["preferred_investment_type"] = "Unknown"
    grouped = customers.merge(grouped, on="customer_id", how="left")
    for c in ["transaction_count","total_transaction_amount","average_transaction_amount","total_investment","average_balance"]:
        grouped[c] = grouped[c].fillna(0)
    grouped["days_since_last_transaction"] = (tx["transaction_date"].max() - grouped["latest_transaction"]).dt.days if tx["transaction_date"].notna().any() else np.nan
    grouped["days_since_last_transaction"] = grouped["days_since_last_transaction"].fillna(-1)
    grouped["latest_transaction"] = grouped["latest_transaction"].astype("string")
    grouped["first_transaction"] = grouped["first_transaction"].astype("string")
    return grouped

def summarize_data(customers, transactions, banks):
    return {
        "customers": int(customers["customer_id"].nunique()),
        "transactions": int(len(transactions)),
        "transaction_value": float(transactions["transaction_amount"].fillna(0).sum()),
        "total_investment": float(transactions["investment_amount"].fillna(0).sum()),
        "date_min": str(transactions["transaction_date"].min()) if transactions["transaction_date"].notna().any() else "Unavailable",
        "date_max": str(transactions["transaction_date"].max()) if transactions["transaction_date"].notna().any() else "Unavailable",
        "bank_branches": int(len(banks)) if banks is not None else 0,
        "missing_transaction_dates": int(transactions["transaction_date"].isna().sum()),
    }

def detect_patterns(features):
    f = features.copy()
    if f.empty: return pd.DataFrame()
    amount_q = f["total_transaction_amount"].quantile(.75)
    count_q = f["transaction_count"].quantile(.75)
    balance_q = f["average_balance"].quantile(.25)
    rows = []
    for _, r in f.iterrows():
        signals = []
        if r["total_transaction_amount"] >= amount_q and amount_q > 0: signals.append("High cumulative transaction value")
        if r["transaction_count"] >= count_q and count_q > 0: signals.append("Frequent transactions")
        if r["average_balance"] <= balance_q and r["average_balance"] > 0: signals.append("Relatively low average balance")
        if r["total_investment"] > 0: signals.append("Existing investment activity")
        if r.get("days_since_last_transaction", -1) >= 90: signals.append("No recent transaction in 90+ days")
        rows.append({"customer_id":r["customer_id"],"signals":", ".join(signals) if signals else "No strong rule-based signal"})
    return pd.DataFrame(rows)
