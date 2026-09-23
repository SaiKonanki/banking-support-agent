"""
generate_mock_data.py

Generates mock data for the Agentic Banking Support Agent portfolio project.

Produces four JSON files in this same directory:
    customers.json
    accounts.json
    transactions.json
    diagnostics.json

No LLM calls, no LangGraph — plain Python data generation only.
This is step 1 of the build (see "Agentic Banking Support Agent - Mock Data
Handoff" doc for the full schema and rationale).
"""

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

random.seed(42)  # reproducible runs

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

FIRST_NAMES = ["Maria", "James", "Aisha", "Wei", "Carlos", "Priya"]
LAST_NAMES = ["Nguyen", "Smith", "Khan", "Chen", "Garcia", "Reddy"]
CITIES = [
    "Austin, TX", "Dallas, TX", "Houston, TX", "San Antonio, TX",
    "Denver, CO", "Phoenix, AZ",
]
BRANCHES = ["Downtown Branch", "Northside Branch", "Westlake Branch", "Online-Only"]
ACCOUNT_TYPES = ["checking", "savings", "credit"]

MERCHANTS = [
    ("Whole Foods Market", "groceries"),
    ("Shell Gas Station", "gas"),
    ("Netflix", "subscription"),
    ("Amazon", "shopping"),
    ("Chipotle", "dining"),
    ("Spotify", "subscription"),
    ("Target", "shopping"),
    ("Starbucks", "dining"),
    ("AT&T", "utilities"),
    ("Delta Air Lines", "travel"),
    ("Planet Fitness", "subscription"),
    ("Best Buy", "shopping"),
    ("Uber", "transport"),
    ("H-E-B", "groceries"),
    ("CVS Pharmacy", "health"),
]

RECURRING_MERCHANTS = {"Netflix", "Spotify", "Planet Fitness", "AT&T"}

TODAY = date(2026, 9, 23)


def random_date(days_back_min=1, days_back_max=45):
    d = TODAY - timedelta(days=random.randint(days_back_min, days_back_max))
    return d.isoformat()


def random_past_date(years_back_min=1, years_back_max=12):
    days_back = random.randint(years_back_min * 365, years_back_max * 365)
    d = TODAY - timedelta(days=days_back)
    return d.isoformat()


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

customers = []
NUM_CUSTOMERS = 6

for i in range(1, NUM_CUSTOMERS + 1):
    customer_id = f"CUST{i:03d}"
    name = f"{FIRST_NAMES[i - 1]} {LAST_NAMES[i - 1]}"
    customers.append({
        "customer_id": customer_id,
        "name": name,
        "phone_number": f"512-555-{1000 + i:04d}",
        "location": random.choice(CITIES),
        "auth_code": f"{random.randint(1000, 9999)}",
    })

# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

accounts = []
account_counter = 1
customer_accounts_map = {}  # customer_id -> list of account_ids

for cust in customers:
    num_accounts = random.randint(1, 3)
    types_for_customer = random.sample(ACCOUNT_TYPES, k=min(num_accounts, len(ACCOUNT_TYPES)))
    customer_accounts_map[cust["customer_id"]] = []

    for acc_type in types_for_customer:
        account_id = f"ACC{account_counter:03d}"
        account_counter += 1
        accounts.append({
            "account_id": account_id,
            "customer_id": cust["customer_id"],
            "account_type": acc_type,
            "branch": random.choice(BRANCHES),
            "date_opened": random_past_date(),
            "active_or_frozen": "active",
            "balance": round(random.uniform(150.0, 12000.0), 2),
        })
        customer_accounts_map[cust["customer_id"]].append(account_id)

# Freeze one account deliberately, to give the account-standing node
# something real to catch.
frozen_account = random.choice(accounts)
frozen_account["active_or_frozen"] = "frozen"

# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

transactions = []
txn_counter = 1

for acc in accounts:
    num_txns = random.randint(5, 10)
    for _ in range(num_txns):
        merchant, category = random.choice(MERCHANTS)
        transaction_id = f"TXN{txn_counter:04d}"
        txn_counter += 1
        amount = (
            round(random.uniform(8.0, 60.0), 2)
            if merchant in RECURRING_MERCHANTS
            else round(random.uniform(5.0, 450.0), 2)
        )
        transactions.append({
            "transaction_id": transaction_id,
            "account_id": acc["account_id"],
            "date": random_date(),
            "location": random.choice(CITIES),
            "amount": amount,
            "description": f"{merchant} - {category}",
            "merchant": merchant,
            "recurring_flag": merchant in RECURRING_MERCHANTS,
        })

# ---------------------------------------------------------------------------
# Seed deliberate cases: duplicates, fraud, failed, pending
# (everything else stays "posted" with no issue, the normal case)
# ---------------------------------------------------------------------------

diagnostics = []
duplicate_pairs = []  # (original_id, duplicate_id)

# --- Duplicates: pick 3 posted, non-recurring transactions and clone them
# a day later on the same account with the same (or near-identical) amount.
candidates_for_dup = [t for t in transactions if not t["recurring_flag"]]
dup_originals = random.sample(candidates_for_dup, k=3)

for orig in dup_originals:
    dup_id = f"TXN{txn_counter:04d}"
    txn_counter += 1
    dup_amount = orig["amount"] if random.random() < 0.7 else round(orig["amount"] + random.uniform(-0.5, 0.5), 2)
    dup_date = (datetime.fromisoformat(orig["date"]) + timedelta(days=1)).date().isoformat()
    duplicate = {
        "transaction_id": dup_id,
        "account_id": orig["account_id"],
        "date": dup_date,
        "location": orig["location"],
        "amount": dup_amount,
        "description": orig["description"],
        "merchant": orig["merchant"],
        "recurring_flag": False,
    }
    transactions.append(duplicate)
    duplicate_pairs.append((orig["transaction_id"], dup_id))

# --- Fraud: 3 transactions flagged, in an unusual location for that customer
fraud_candidates = [t for t in transactions if t["transaction_id"] not in
                     [d for pair in duplicate_pairs for d in pair]]
fraud_txns = random.sample(fraud_candidates, k=3)
fraud_ids = {t["transaction_id"] for t in fraud_txns}
for t in fraud_txns:
    t["location"] = random.choice([c for c in CITIES if c != t["location"]])

# --- Failed: 2 transactions, one insufficient_funds, one card_frozen
remaining = [t for t in transactions if t["transaction_id"] not in fraud_ids
             and t["transaction_id"] not in [d for pair in duplicate_pairs for d in pair]]
failed_txns = random.sample(remaining, k=2)
failed_reasons = ["insufficient_funds", "card_frozen"]

# --- Pending: 2 more transactions
remaining2 = [t for t in remaining if t not in failed_txns]
pending_txns = random.sample(remaining2, k=2)

special_ids = {t["transaction_id"] for t in fraud_txns} \
    | {t["transaction_id"] for t in failed_txns} \
    | {t["transaction_id"] for t in pending_txns} \
    | {d for pair in duplicate_pairs for d in pair}

# Build diagnostics for every transaction
dup_id_lookup = {}
for orig_id, dup_id in duplicate_pairs:
    dup_id_lookup[orig_id] = dup_id
    dup_id_lookup[dup_id] = orig_id

for t in transactions:
    tid = t["transaction_id"]
    if tid in dup_id_lookup:
        diagnostics.append({
            "transaction_id": tid,
            "transaction_status": "posted",
            "block_reason": "none",
            "fraud_flag": False,
            "duplicate_match_id": dup_id_lookup[tid],
        })
    elif tid in fraud_ids:
        diagnostics.append({
            "transaction_id": tid,
            "transaction_status": "posted",
            "block_reason": "none",
            "fraud_flag": True,
            "duplicate_match_id": "none",
        })
    elif t in failed_txns:
        reason = failed_reasons[failed_txns.index(t)]
        diagnostics.append({
            "transaction_id": tid,
            "transaction_status": "failed",
            "block_reason": reason,
            "fraud_flag": False,
            "duplicate_match_id": "none",
        })
    elif t in pending_txns:
        diagnostics.append({
            "transaction_id": tid,
            "transaction_status": "pending",
            "block_reason": "none",
            "fraud_flag": False,
            "duplicate_match_id": "none",
        })
    else:
        diagnostics.append({
            "transaction_id": tid,
            "transaction_status": "posted",
            "block_reason": "none",
            "fraud_flag": False,
            "duplicate_match_id": "none",
        })

# ---------------------------------------------------------------------------
# Write out JSON files
# ---------------------------------------------------------------------------

with open(OUT_DIR / "customers.json", "w") as f:
    json.dump(customers, f, indent=2)

with open(OUT_DIR / "accounts.json", "w") as f:
    json.dump(accounts, f, indent=2)

with open(OUT_DIR / "transactions.json", "w") as f:
    json.dump(transactions, f, indent=2)

with open(OUT_DIR / "diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)

print(f"Generated {len(customers)} customers")
print(f"Generated {len(accounts)} accounts (1 frozen: {frozen_account['account_id']})")
print(f"Generated {len(transactions)} transactions")
print(f"  - {len(duplicate_pairs)} duplicate pairs: {duplicate_pairs}")
print(f"  - {len(fraud_ids)} fraud-flagged: {sorted(fraud_ids)}")
print(f"  - {len(failed_txns)} failed: {[t['transaction_id'] for t in failed_txns]}")
print(f"  - {len(pending_txns)} pending: {[t['transaction_id'] for t in pending_txns]}")
print(f"Generated {len(diagnostics)} diagnostics records")
print("Done. Files written to:", OUT_DIR)
