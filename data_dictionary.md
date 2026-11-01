# Data dictionary - Default of Credit Card Clients (UCI id 350)

Source: UCI Machine Learning Repository, id 350, DOI `10.24432/C55S3H`, CC BY 4.0.
Introductory paper: Yeh, I., & Lien, C. (2009).

Provenance: the CSV was exported from UCI id 350 with the `ucimlrepo` package, row for row,
with the values unmodified; the column names follow the UCI codebook (`X1` to `X23`). The
target column is named `default`.

`30,000` clients, `23` explanatory variables plus the target. The saved CSV is
`default_of_credit_card_clients.csv`.

| Column | Meaning | Coding |
|---|---|---|
| X1 | LIMIT_BAL: amount of the given credit (NT dollar), includes individual and family credit | integer |
| X2 | SEX | 1 = male, 2 = female |
| X3 | EDUCATION | 1 = graduate school, 2 = university, 3 = high school, 4 = others |
| X4 | MARRIAGE | 1 = married, 2 = single, 3 = others |
| X5 | AGE | years |
| X6 - X11 | repayment status in September, August, ... April 2005 | -2 = no consumption / balance paid in full, -1 = pay duly, 0 = no consumption / revolving credit, 1 = one-month delay, 2 = two-month delay, and so on; code `9` does not occur in this file, observed range `-2` to `8` |
| X12 - X17 | bill statement amount in September, ... April 2005 | NT dollar |
| X18 - X23 | amount of previous payment in September, ... April 2005 | NT dollar |
| default | target: default payment next month | 1 = yes, 0 = no |

Known coding quirks (reported by the source page): `EDUCATION` has undocumented
values `0`, `5`, and `6`, and `MARRIAGE` has an undocumented `0`; these are folded
into `others` during preprocessing.
