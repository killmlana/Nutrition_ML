# load trained model + scalar, accept manual input, output prediction 

import joblib
import numpy as np
from src.recommend import get_recommendation

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
    probability = model.predict_proba(sample_scaled).max()

    print("\nPredicted Class:", prediction)
    print("Confidence:", round(probability * 100, 2), "%")
    print("Recommendation:", get_recommendation(prediction))
