import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess():
    df = pd.read_csv("data/raw/synthetic_data.csv")

    X = df[["age", "sex", "weight", "height", "muac", "hb", "bmi"]]

    y_acute = df["acute_label"]
    y_stunting = df["stunting_flag"]

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

    print("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
