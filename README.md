# Customer Churn Analysis (EDA & Storytelling)

A portfolio-ready exploratory data analysis of why subscription customers cancel,
with behavioral segmentation and a data-backed retention strategy.

## Problem
Acquiring customers is expensive, so retention matters. This project finds **which**
customers churn and **why**, then turns the findings into action.

## Key insight (from the included synthetic data)
Month-to-month customers on fiber internet with no tech support churn at roughly
**3x the company baseline** of ~27%.

## Project structure
```
churn-analysis/
├── data/                    # dataset lands here
├── notebooks/
│   └── churn_eda.ipynb      # same analysis, notebook form
├── src/
│   ├── generate_data.py     # creates a realistic synthetic dataset
│   └── churn_eda.py         # full EDA + segmentation + story
├── outputs/
│   ├── figures/             # generated charts (.png)
│   ├── churn_segments.csv   # ranked high-risk segments
│   └── insights.md          # auto-generated story
├── requirements.txt
└── README.md
```

## Quickstart
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py       # writes data/telco_churn.csv
python src/churn_eda.py           # writes charts + insights to outputs/
```

## Use the real dataset instead
Download the **Telco Customer Churn** dataset from Kaggle
(`WA_Fn-UseC_-Telco-Customer-Churn.csv`), save it as `data/telco_churn.csv`,
and skip `generate_data.py`. The cleaning code already handles the real file's
`TotalCharges` blank-string quirk.

## Approach
1. **Clean & engineer** - fix types, normalize categories, build tenure groups & service counts
2. **EDA** - baseline churn, churn-by-category, numeric drivers, correlation
3. **Segment** - rank customer groups by churn risk multiple
4. **Story** - headline insight + retention strategy tied to revenue at risk

## Tools
Python · pandas · numpy · matplotlib · seaborn · scikit-learn

## Stretch goals
- Predictive model (logistic regression / random forest) with feature importance
- Power BI dashboard fed by `outputs/churn_segments.csv`
- Streamlit what-if simulator
