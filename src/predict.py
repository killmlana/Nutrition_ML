import joblib
import pandas as pd
from src.rule_engine import check_anemia
from src.recommend import acute_text, build_recommendation


def predict_child():
    acute_model = joblib.load("models/acute_model.pkl")
    stunting_model = joblib.load("models/stunting_model.pkl")
    scaler = joblib.load("data/processed/scaler.pkl")

    print("\nEnter Child Data:")
    age = int(input("Age (months): "))
    sex = int(input("Sex (0=Female, 1=Male): "))
    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))
    muac = float(input("MUAC (mm): "))
    hb = float(input("Hemoglobin: "))

    bmi = weight / ((height / 100) ** 2)

    sample_df = pd.DataFrame(
        [[age, sex, weight, height, muac, hb, bmi]],
        columns=["age", "sex", "weight", "height", "muac", "hb", "bmi"]
    )

    sample_scaled = scaler.transform(sample_df)

    acute_pred = acute_model.predict(sample_scaled)[0]
    stunting_pred = stunting_model.predict(sample_scaled)[0]
    anemia_flag = check_anemia(hb)

    print("\n--- Assessment Report ---")
    print("Acute Malnutrition:", acute_text(acute_pred))
    print("Stunting Risk:", "YES" if stunting_pred else "NO")
    print("Anemia:", "YES" if anemia_flag else "NO")

    print("\n--- Dietary Focus ---")
    for rec in build_recommendation(acute_pred, stunting_pred, anemia_flag):
        print("✓", rec)


if __name__ == "__main__":
    predict_child()
