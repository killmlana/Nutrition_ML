import joblib
import pandas as pd
from src.rule_engine import check_anemia, check_muac
from src.recommend import acute_text, muac_text, risk_level, build_recommendation


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

    wasting_pred = int(acute_model.predict(sample_scaled)[0])
    stunting_pred = int(stunting_model.predict(sample_scaled)[0])
    anemia_flag = int(check_anemia(hb))
    muac_status = check_muac(muac)
    risk = risk_level(wasting_pred, stunting_pred, anemia_flag, muac_status)

    print("\n--- Assessment Report ---")
    print("Wasting (WHZ):", acute_text(wasting_pred))
    print("MUAC Status:", muac_text(muac_status))
    print("Stunting:", "Yes" if stunting_pred else "No")
    print("Anemia:", "Yes" if anemia_flag else "No")
    print("Risk Level:", risk.upper())

    print("\n--- Recommendations ---")
    for rec in build_recommendation(wasting_pred, stunting_pred, anemia_flag, muac_status):
        print(f"\n[{rec['priority'].upper()}] {rec['category']}")
        print(f"  {rec['text']}")
        for detail in rec.get('details', []):
            print(f"  - {detail}")


if __name__ == "__main__":
    predict_child()
