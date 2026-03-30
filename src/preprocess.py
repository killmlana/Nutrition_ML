import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess():
    df = pd.read_stata("data/synthetic_cnns_multistate_1to4_from_factsheets.dta")

    df["sex"] = df["sex"].map({"Female": 0, "Male": 1})
    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

    X = df[["age_months", "sex", "weight_kg", "height_cm", "muac_mm", "hemoglobin_g_dl", "bmi"]]
    X.columns = ["age", "sex", "weight", "height", "muac", "hb", "bmi"]

    y_acute = df["wasted_proxy"]
    y_stunting = df["stunted_proxy"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_acute_train, y_acute_test = train_test_split(
        X_scaled, y_acute, test_size=0.2, random_state=42
    )

    _, _, y_stunting_train, y_stunting_test = train_test_split(
        X_scaled, y_stunting, test_size=0.2, random_state=42
    )

    os.makedirs("data/processed", exist_ok=True)

    joblib.dump(
        (X_train, X_test, y_acute_train, y_acute_test),
        "data/processed/acute_data.pkl"
    )

    joblib.dump(
        (X_train, X_test, y_stunting_train, y_stunting_test),
        "data/processed/stunting_data.pkl"
    )

    joblib.dump(scaler, "data/processed/scaler.pkl")

    print(f"Preprocessing complete. {len(df)} samples from CNNS data.")


if __name__ == "__main__":
    preprocess()
