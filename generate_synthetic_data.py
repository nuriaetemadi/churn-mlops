"""
Synthetic Telco Customer Churn Data Generator
Group 7: Pablo Infante · Nuria Etemadi · Tenaw Belete · Jan Wilhelm · Selim El Khoury

Generates realistic synthetic data matching the exact schema of the
WA_Fn-UseC_-Telco-Customer-Churn.csv dataset.

Usage:
  python generate_synthetic_data.py --n 500 --output data/synthetic/synthetic_customers.csv
  python generate_synthetic_data.py --n 100 --churn_rate 0.5  # force higher churn
"""

import argparse
import logging
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def generate_customer_id() -> str:
    """Generate a fake customerID in the same format: XXXX-XXXXX."""
    part1 = "".join(random.choices(string.digits, k=4))
    part2 = "".join(random.choices(string.ascii_uppercase, k=5))
    return f"{part1}-{part2}"


def generate_synthetic_data(n: int = 500, churn_rate: float = 0.265,
                             random_seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate n synthetic customers with realistic correlations:
      - Month-to-month contracts → higher churn probability
      - Short tenure → higher churn probability
      - No online security / no tech support → higher churn
      - High monthly charges with month-to-month → highest churn

    Parameters
    ----------
    n           : number of customers to generate
    churn_rate  : approximate target churn rate (real dataset ≈ 0.265)
    random_seed : for reproducibility
    """
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)

    # ── Static distributions from real dataset ────────────────────────────
    genders = rng.choice(["Male", "Female"], size=n, p=[0.505, 0.495])
    senior = rng.choice([0, 1], size=n, p=[0.838, 0.162])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.483, 0.517])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.298, 0.702])
    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.904, 0.096])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.592, 0.408])

    # Contract type – strongly influences churn
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.550, 0.210, 0.240]
    )

    # Tenure: correlated with contract
    tenure = np.zeros(n, dtype=int)
    for i, c in enumerate(contract):
        if c == "Month-to-month":
            tenure[i] = int(rng.integers(0, 36))
        elif c == "One year":
            tenure[i] = int(rng.integers(6, 60))
        else:
            tenure[i] = int(rng.integers(12, 72))

    # Internet service
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n, p=[0.343, 0.440, 0.217]
    )

    # Internet-dependent services
    def internet_dep(rng, n, internet, yes_p=0.5):
        result = []
        for i in range(n):
            if internet[i] == "No":
                result.append("No internet service")
            else:
                result.append(rng.choice(["Yes", "No"], p=[yes_p, 1 - yes_p]))
        return np.array(result)

    online_security = internet_dep(rng, n, internet, yes_p=0.29)
    online_backup = internet_dep(rng, n, internet, yes_p=0.34)
    device_protection = internet_dep(rng, n, internet, yes_p=0.34)
    tech_support = internet_dep(rng, n, internet, yes_p=0.29)
    streaming_tv = internet_dep(rng, n, internet, yes_p=0.38)
    streaming_movies = internet_dep(rng, n, internet, yes_p=0.39)

    # Multiple lines
    multiple_lines = []
    for i in range(n):
        if phone_service[i] == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"], p=[0.422, 0.578]))
    multiple_lines = np.array(multiple_lines)

    # Payment method
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.336, 0.229, 0.218, 0.217]
    )

    # Monthly charges – depends on internet service
    monthly_charges = np.zeros(n)
    for i in range(n):
        base = {"No": rng.uniform(18, 30),
                "DSL": rng.uniform(25, 65),
                "Fiber optic": rng.uniform(70, 110)}[internet[i]]
        monthly_charges[i] = round(float(base), 2)

    total_charges = np.array([
        round(float(tenure[i] * monthly_charges[i] + rng.uniform(-5, 5)), 2)
        if tenure[i] > 0 else 0.0
        for i in range(n)
    ])
    total_charges = np.clip(total_charges, 0, None)

    # ── Churn label with realistic correlations ───────────────────────────
    churn_prob = np.full(n, 0.10)  # base rate

    # Contract type effect
    churn_prob += np.where(contract == "Month-to-month", 0.25, 0.0)
    churn_prob += np.where(contract == "Two year", -0.08, 0.0)

    # Tenure effect (new customers churn more)
    churn_prob += np.where(tenure < 6, 0.15, 0.0)
    churn_prob += np.where(tenure > 36, -0.10, 0.0)

    # Fiber optic – historically higher churn in dataset
    churn_prob += np.where(internet == "Fiber optic", 0.08, 0.0)

    # No online security
    churn_prob += np.where(online_security == "No", 0.06, 0.0)

    # Electronic check – correlates with churn
    churn_prob += np.where(payment == "Electronic check", 0.07, 0.0)

    # High monthly charges
    churn_prob += np.where(monthly_charges > 85, 0.05, 0.0)

    # Clamp to [0, 1]
    churn_prob = np.clip(churn_prob, 0.0, 1.0)

    # Scale to match target churn_rate
    scale = churn_rate / churn_prob.mean()
    churn_prob = np.clip(churn_prob * scale, 0.0, 1.0)

    churn = rng.binomial(1, churn_prob).astype(bool)
    churn_label = np.where(churn, "Yes", "No")

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "customerID": [generate_customer_id() for _ in range(n)],
        "gender": genders,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
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
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_label,
    })

    actual_rate = churn.mean()
    logger.info(
        "Generated %d synthetic customers | Churn rate: %.1f%% (target %.1f%%)",
        n, actual_rate * 100, churn_rate * 100
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic churn data")
    parser.add_argument("--n", type=int, default=500, help="Number of customers")
    parser.add_argument("--churn_rate", type=float, default=0.265,
                        help="Approximate churn rate (0–1)")
    parser.add_argument("--output", type=Path,
                        default=Path("data/synthetic/synthetic_customers.csv"))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(n=args.n, churn_rate=args.churn_rate, random_seed=args.seed)
    df.to_csv(args.output, index=False)
    logger.info("Saved to %s", args.output)
    print(df.head())
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")


if __name__ == "__main__":
    main()
