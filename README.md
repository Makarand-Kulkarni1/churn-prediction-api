# Customer Churn Prediction API

An end-to-end machine learning project that predicts customer churn risk for a telecom company and serves predictions through a deployed REST API — built to demonstrate the full pipeline from raw data to a usable, live service, not just a model in a notebook.

## Overview

Customer churn is one of the most direct revenue problems a subscription-based business faces. This project builds a model that flags at-risk customers *before* they leave, so a retention team can intervene proactively rather than reactively. The dataset used is IBM's Telco Customer Churn dataset (7,043 customers) — a synthetic-but-realistic industry-standard benchmark widely used in published churn research.

## What Was Built

- **Data cleaning:** identified and fixed a hidden data-quality issue where `TotalCharges` contained disguised missing values (blank strings, not proper `NaN`) — a bug that would silently pass a basic `.isnull()` check.
- **Feature engineering:** encoded categorical variables deliberately — binary label encoding for true Yes/No fields, one-hot encoding for unordered multi-category fields — to avoid introducing false numeric ordering into the model.
- **Modeling:** trained an XGBoost classifier, achieving **0.837 ROC-AUC** on a held-out test set.
- **Threshold tuning:** rather than accepting the default 0.5 classification cutoff, I evaluated the precision/recall tradeoff explicitly and chose **0.35** — since for this business case, missing an actual churner (a false negative) is more costly than a wasted retention offer (a false positive). This raised recall from 53% to 69%.
- **Feature importance:** contract type (`Month-to-month`) and internet service type (`Fiber optic`) emerged as by far the dominant churn drivers — a concrete, actionable insight for a retention strategy, not just a prediction.
- **Deployment:** wrapped the trained model in a FastAPI service with a `/predict` endpoint, input validation via Pydantic, and interactive Swagger documentation — deployed live on Render.

## Tech Stack
Python · pandas · scikit-learn · XGBoost · FastAPI · Pydantic · Render

## Live Demo
API docs: `https://churn-prediction-api-7jvh.onrender.com/docs`

## Limitations
This dataset is synthetic (IBM-generated for benchmarking), so results reflect realistic *correlational* patterns rather than proven causal drivers — a production deployment would benefit from real usage and support-interaction data to validate why customers actually leave.

## Run Locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive API.
