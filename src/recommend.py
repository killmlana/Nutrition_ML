# maps predicted class and makes recommendatiions

LABELS = {
    0: "Normal",
    1: "Moderate Acute Malnutrition (MAM)",
    2: "Severe Acute Malnutrition (SAM)",
    3: "Stunting Risk",
    4: "Anemia"
}

RECOMMENDATIONS = {
    0: "Maintain balanced diet with vegetables, pulses, milk and fruits.",
    1: "Increase protein and calorie intake: lentils, banana, oil, peanut paste.",
    2: "Severe risk detected. Provide high-protein, energy-dense foods immediately and refer to clinic.",
    3: "Focus on long-term protein and micronutrient-rich foods.",
    4: "Increase iron-rich foods: spinach, ragi, jaggery + Vitamin C fruits."
}

def interpret_prediction(label):
    return LABELS.get(label, "Unknown Condition")

def get_recommendation(label):
    return RECOMMENDATIONS.get(label, "No recommendation available.")
