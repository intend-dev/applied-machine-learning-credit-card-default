# Applied Machine Learning with Python: Credit Card Default Prediction

## Overview

This project builds and tunes a model that predicts next-month credit card default, using the UCI Default of Credit Card Clients dataset (Yeh & Lien, 2009; Yeh, 2009). The tuned random forest reaches ROC AUC `0.7799` on a held-out test set, and a cost-tuned cut-off cuts the expected bank loss from `4,448` to `3,347` units.
Repository: https://github.com/intend-dev/applied-machine-learning-credit-card-default

## Repository

- `modeling.ipynb`: executed notebook with preprocessing, model comparison, tuning, evaluation, fairness checks and figures
- `module_summary.md` and `module_summary.pdf`: the report and print-ready report
- `figures/`: two PNG figures from the notebook
- `default_of_credit_card_clients.csv`: UCI Default of Credit Card Clients data
- `requirements.txt`: pinned package versions
- `scripts/fetch_dataset.py`: one-time UCI fetch that produced the CSV
- `build/build_report.py`: builds the PDF report from the Markdown

## Reproduce

`requirements.txt` was generated with `pip freeze`, so install its pinned packages with `pip install -r requirements.txt` in an activated Python 3.13 environment.
The `uv` commands below show the same setup with `uv`, the alternative used for this rebuild:

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace modeling.ipynb
```

If you regenerate `requirements.txt`, avoid PowerShell `>` because it writes UTF-16.
The notebook reads `default_of_credit_card_clients.csv` from the same folder and needs neither network access nor credentials.

## Dataset

- [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) from the UCI Machine Learning Repository
- `30,000` rows and `24` columns
- Target `default`: `6,636` defaults (`22.12%` of the sample)
- DOI: https://doi.org/10.24432/C55S3H
- `default_of_credit_card_clients.csv` checksum: `FB9A577241F130FF09F90995C0C15A58A612D52FC162CBC5EC823E23B7B876A5`

## License

Shared for portfolio and review purposes only, no reuse license is granted, and no license file is included.
The dataset is public data under its own terms (CC BY 4.0).
