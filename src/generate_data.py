import numpy as np
import pandas as pd
import os
import random


def generate_data(n=5000):
    data = []

    for _ in range(n):
        age = random.randint(6, 59)
        sex = random.randint(0, 1)

        # Simulated growth formulas
        expected_height = (age * 0.5) + 50
        height = np.random.normal(expected_height, 4)

        expected_weight = age * 0.25 + 4
        weight = np.random.normal(expected_weight, 1.2)

        muac = np.random.normal(135, 12)
        hb = np.random.normal(11.5, 1.2)

        bmi = weight / ((height / 100) ** 2)

        # --- Acute Malnutrition ---
        if muac < 115:
            acute_label = 2  # SAM
        elif muac < 125:
            acute_label = 1  # MAM
        else:
            acute_label = 0  # Normal

        # --- Stunting ---
        stunting_flag = 1 if height < 0.9 * expected_height else 0

        # --- Anemia ---
        anemia_flag = 1 if hb < 11 else 0

        data.append([
            age, sex, weight, height, muac, hb, bmi,
            acute_label, stunting_flag, anemia_flag
        ])

    columns = [
        "age", "sex", "weight", "height", "muac", "hb", "bmi",
        "acute_label", "stunting_flag", "anemia_flag"
    ]

    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_data()
    df.to_csv("data/raw/synthetic_data.csv", index=False)
    print("Multi-label synthetic dataset generated.")
