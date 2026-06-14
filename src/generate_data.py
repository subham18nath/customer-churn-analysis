"""Generate a synthetic Telco-style customer churn dataset.

This lets you run the whole project immediately without downloading
anything from Kaggle. The churn patterns are intentionally realistic:
short-tenure, month-to-month, fiber, electronic-check, and
no-tech-support customers churn more.

Usage:
    python src/generate_data.py
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 7043  # same size as the classic Telco dataset


def pick(options, probs):
    return RNG.choice(options, size=N, p=probs)


def build_dataframe() -> pd.DataFrame:
    gender = pick(["Male", "Female"], [0.50, 0.50])
    senior = pick([0, 1], [0.84, 0.16])
    partner = pick(["Yes", "No"], [0.48, 0.52])
    dependents = pick(["Yes", "No"], [0.30, 0.70])

    contract = pick(
        ["Month-to-month", "One year", "Two year"],
        [0.55, 0.21, 0.24],
    )
    internet = pick(
        ["Fiber optic", "DSL", "No"],
        [0.44, 0.34, 0.22],
    )
    payment = pick(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.23, 0.22, 0.21],
    )
    paperless = pick(["Yes", "No"], [0.59, 0.41])
    phone = pick(["Yes", "No"], [0.90, 0.10])

    def service_col(has_internet_dependency=True):
        vals = RNG.choice(["Yes", "No"], size=N, p=[0.40, 0.60]).astype(object)
        if has_internet_dependency:
            vals[internet == "No"] = "No internet service"
        return vals

    online_security = service_col()
    online_backup = service_col()
    device_protection = service_col()
    tech_support = service_col()
    streaming_tv = service_col()
    streaming_movies = service_col()

    multiple_lines = RNG.choice(["Yes", "No"], size=N, p=[0.42, 0.58]).astype(object)
    multiple_lines[phone == "No"] = "No phone service"

    # Tenure: month-to-month customers skew newer
    tenure = np.where(
        contract == "Month-to-month",
        RNG.integers(0, 50, N),
        RNG.integers(10, 73, N),
    ).astype(int)

    # Monthly charges driven by internet type + add-ons
    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 50, 20)).astype(float)
    addons = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    monthly = base + addons * 5 + RNG.normal(0, 4, N)
    monthly = np.round(np.clip(monthly, 18, 120), 2)

    total = np.round(monthly * tenure + RNG.normal(0, 20, N), 2)
    total = np.clip(total, 0, None)
    # Mimic the real dataset's quirk: new customers have blank TotalCharges
    total_str = total.astype(object)
    total_str[tenure == 0] = " "

    # --- Churn probability model ---
    logit = -2.2
    logit += np.where(contract == "Month-to-month", 1.6, np.where(contract == "One year", 0.1, -1.2))
    logit += np.where(internet == "Fiber optic", 0.9, np.where(internet == "DSL", 0.1, -0.7))
    logit += np.where(tech_support == "No", 0.5, 0.0)
    logit += np.where(online_security == "No", 0.4, 0.0)
    logit += np.where(payment == "Electronic check", 0.6, 0.0)
    logit += np.where(paperless == "Yes", 0.25, 0.0)
    logit += np.where(senior == 1, 0.35, 0.0)
    logit += -0.03 * tenure
    logit += 0.012 * (monthly - monthly.mean())
    logit += RNG.normal(0, 0.4, N)

    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(RNG.random(N) < prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-CUST" for i in range(N)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_str,
            "Churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    df = build_dataframe()
    df.to_csv("data/telco_churn.csv", index=False)
    print(f"Wrote data/telco_churn.csv  shape={df.shape}")
    print(f"Overall churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
