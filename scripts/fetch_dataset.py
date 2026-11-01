"""Fetch the UCI Default of Credit Card Clients dataset (id 350).

One-time fetch; the resulting CSV is committed so a run needs no network.
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
DOI: 10.24432/C55S3H  (CC BY 4.0)
"""
from ucimlrepo import fetch_ucirepo

repo = fetch_ucirepo(id=350)
X = repo.data.features
y = repo.data.targets
df = X.copy()
# The pinned bytes, the notebook, and the dictionary all name the target
# 'default'; the upstream metadata name is not stable ('Y' on 2026-11-01).
df["default"] = y.iloc[:, 0].values
df.to_csv("default_of_credit_card_clients.csv", index=False)
print(f"Fetched {len(df)} rows, {len(df.columns)} columns")