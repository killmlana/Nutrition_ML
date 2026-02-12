# load trained model + scaler, accept manual input, output prediction 

import joblib
import numpy as np
from src.recommend import interpret_prediction, get_recommendation

def predict_child():
    model = joblib.load("models/malnutrition_model.pkl")
    scaler = joblib.load("data/processed/scaler.pkl")

    print("\nEnter Child Data:")
    age = int(input("Age (months): "))
    sex = int(input("Sex (0=Female, 1=Male): "))
    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))
    muac = float(input("MUAC (mm): "))
    hb = float(input("Hemoglobin: "))

    bmi = weight / ((height/100)**2)

    sample = np.array([[age, sex, weight, height, muac, hb, bmi]])
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    confidence = max(probabilities)

    condition = interpret_prediction(prediction)
    recommendation = get_recommendation(prediction)

    print("\n--- Input Summary ---")
    print(f"Age: {age} months")
    print(f"Weight: {weight} kg")
    print(f"Height: {height} cm")
    print(f"MUAC: {muac} mm")
    print(f"Hemoglobin: {hb}")

    print("\n--- Assessment Report ---")
    print(f"Condition Detected: {condition}")
    print(f"Model Confidence: {round(confidence * 100, 2)}%")

    print("\n--- Dietary Recommendation ---")
    print(recommendation)


if __name__ == "__main__":
    predict_child()
