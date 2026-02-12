def acute_text(label):
    mapping = {
        0: "Normal",
        1: "Moderate Acute Malnutrition (MAM)",
        2: "Severe Acute Malnutrition (SAM)"
    }
    return mapping.get(label, "Unknown")


def build_recommendation(acute_label, stunting_flag, anemia_flag):
    recommendations = []

    if acute_label == 1:
        recommendations.append("Increase protein and calorie intake.")
    if acute_label == 2:
        recommendations.append("Provide high-protein, energy-dense foods immediately and refer to clinic.")

    if stunting_flag:
        recommendations.append("Focus on long-term protein and micronutrient-rich foods.")

    if anemia_flag:
        recommendations.append("Increase iron-rich foods + Vitamin C fruits.")

    if not recommendations:
        recommendations.append("Maintain balanced diet with vegetables, pulses, milk and fruits.")

    return recommendations
