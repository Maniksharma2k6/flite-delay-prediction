# Flight Delay Prediction

**Machine Learning Mini Project** | Manik Sharma | Roll No: CS3236 | TY Div B

## Problem Statement

Predict *before take-off* whether a flight will arrive more than 15 minutes late
at its destination.  Binary classification (delayed = 1, on-time = 0).

## Dataset

- **Source:** nycflights13 (Bureau of Transportation Statistics) + Iowa Environmental
  Mesonet hourly weather.
- **Rows:** 12,000 sampled flights departing New York in 2013.
- **Features:** 15 (flight info + departure weather).
- **Target:** `delayed` (1 = >15 min late, 0 = on time).  ~23.7% positive.

## Algorithms

| # | Algorithm | Key Settings |
|---|-----------|-------------|
| 1 | Logistic Regression | C=0.1, balanced |
| 2 | K-Nearest Neighbours | k=5, distance-weighted |
| 3 | Support Vector Machine | RBF kernel, C=1, balanced |
| 4 | Random Forest | 200 trees, min_leaf=5, balanced |

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder Structure

```
app.py                      # Home page
pages/
  1_Train_Data.py            # Train page
  2_Test_Data.py             # Test page
  3_Algorithm_Report.py      # Report page
utils.py                    # Shared utilities
data/flights_sample.csv     # Dataset
models/                     # Runtime-created (models + holdout test set)
requirements.txt
README.md
```
