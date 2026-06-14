"""Customer Churn Analysis - end-to-end EDA, segmentation, and storytelling.

Run this after generating the dataset:
    python src/generate_data.py
    python src/churn_eda.py

Outputs:
    outputs/figures/*.png    - all charts
    outputs/churn_segments.csv - ranked high-risk segments
    outputs/insights.md       - the auto-generated story
"""
import os

import matplotlib

matplotlib.use("Agg")  # render to files without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 1 - Load
# ---------------------------------------------------------------------------
def load_data(path="data/telco_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Phase 2 - Clean & engineer
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])

    for col in df.select_dtypes("object").columns:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"],
    )

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    df["charges_per_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


# ---------------------------------------------------------------------------
# Phase 3 - EDA
# ---------------------------------------------------------------------------
def churn_by(df, col, baseline):
    g = (
        df.groupby(col, observed=True)["Churn"]
        .agg(churn_rate="mean", customers="size")
        .sort_values("churn_rate", ascending=False)
    )
    g["vs_baseline"] = g["churn_rate"] - baseline
    return g


def run_eda(df: pd.DataFrame, baseline: float):
    # Balance
    plt.figure()
    sns.countplot(data=df, x="Churn")
    plt.title("Churned (1) vs Retained (0)")
    plt.savefig(f"{FIG_DIR}/churn_balance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Category bars
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(
        axes.flat, ["Contract", "InternetService", "PaymentMethod", "tenure_group"]
    ):
        (
            df.groupby(col, observed=True)["Churn"].mean()
            .sort_values()
            .plot(kind="barh", ax=ax, color="#d1495b")
        )
        ax.axvline(baseline, ls="--", color="gray", label="baseline")
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("Churn rate")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Numeric drivers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False, ax=axes[0])
    axes[0].set_title("Tenure: churned vs retained")
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, ax=axes[1])
    axes[1].set_title("Monthly charges: churned vs retained")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/numeric_drivers.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    num_df = df.select_dtypes(include=["number"])
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation of numeric features")
    plt.savefig(f"{FIG_DIR}/correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nChurn by Contract:")
    print(churn_by(df, "Contract", baseline).round(3))
    print("\nChurn by PaymentMethod:")
    print(churn_by(df, "PaymentMethod", baseline).round(3))


# ---------------------------------------------------------------------------
# Phase 4 - Segmentation
# ---------------------------------------------------------------------------
def segment(df: pd.DataFrame, baseline: float) -> pd.DataFrame:
    segments = (
        df.groupby(["Contract", "InternetService", "TechSupport"], observed=True)["Churn"]
        .agg(churn_rate="mean", customers="size")
        .reset_index()
    )
    segments = segments[segments["customers"] >= 100].copy()
    segments["risk_multiple"] = segments["churn_rate"] / baseline
    segments = segments.sort_values("churn_rate", ascending=False)
    os.makedirs("outputs", exist_ok=True)
    segments.to_csv("outputs/churn_segments.csv", index=False)
    return segments


# ---------------------------------------------------------------------------
# Phase 5 - Storytelling
# ---------------------------------------------------------------------------
def tell_story(df, baseline, segments):
    top = segments.iloc[0]
    seg = df[
        (df["Contract"] == top["Contract"])
        & (df["InternetService"] == top["InternetService"])
        & (df["TechSupport"] == top["TechSupport"])
    ]
    at_risk = seg["MonthlyCharges"].sum() * 12
    saved = seg[seg["Churn"] == 1]["MonthlyCharges"].sum() * 12 * 0.25

    def to_md_table(frame):
        cols = list(frame.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(v) for v in rec) + " |"
            for rec in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *rows])

    lines = [
        "# Churn Analysis - Key Insights\n",
        f"**Overall churn rate (baseline):** {baseline:.1%}\n",
        "## Headline insight\n",
        f"Customers on **{top['Contract']}** contracts with "
        f"**{top['InternetService']}** internet and "
        f"**TechSupport = {top['TechSupport']}** churn at "
        f"**{top['churn_rate']:.1%}** - about **{top['risk_multiple']:.1f}x** the baseline.\n",
        "## Top 5 high-risk segments\n",
        to_md_table(segments.head(5).round(3)),
        "\n## Revenue at stake (top segment)\n",
        f"- Annual revenue exposed: **${at_risk:,.0f}**",
        f"- Revenue saved with a 25% churn reduction: **${saved:,.0f}**\n",
        "## Retention strategy\n",
        "1. Migrate month-to-month customers to annual contracts with a discount.",
        "2. Launch a 90-day onboarding journey to protect first-year customers.",
        "3. Bundle a free tech-support trial for at-risk fiber customers.",
        "4. Incentivize auto-pay to move users off electronic check.",
        "5. Proactive quality outreach for high-paying fiber customers.",
    ]
    with open("outputs/insights.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:6]))
    print("\nWrote outputs/insights.md and outputs/churn_segments.csv")


def main():
    df = load_data()
    df = clean(df)
    baseline = df["Churn"].mean()
    print(f"Overall churn rate: {baseline:.1%}")
    run_eda(df, baseline)
    segments = segment(df, baseline)
    tell_story(df, baseline, segments)
    print("\nDone. See the outputs/ folder.")


if __name__ == "__main__":
    main()
