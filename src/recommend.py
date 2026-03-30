def acute_text(label):
    return {0: "Normal", 1: "Wasted"}.get(label, "Unknown")


def muac_text(muac_status):
    return {"sam": "SAM (MUAC < 115mm)", "mam": "MAM (MUAC < 125mm)", "normal": "Normal"}.get(muac_status, "Unknown")


def risk_level(wasting, stunting, anemia, muac_status="normal"):
    # MUAC SAM is always critical
    if muac_status == "sam":
        return "critical"
    # MUAC MAM or ML-predicted wasting combined with other conditions
    has_acute = wasting or muac_status == "mam"
    conditions = sum([bool(has_acute), bool(stunting), bool(anemia)])
    if has_acute and conditions >= 2:
        return "critical"
    if has_acute or (stunting and anemia):
        return "high"
    if stunting or anemia:
        return "moderate"
    return "low"


def build_recommendation(wasting, stunting, anemia, muac_status="normal"):
    recs = []

    # MUAC-based acute malnutrition (clinical rule — takes priority)
    if muac_status == "sam":
        recs.append({
            "priority": "urgent",
            "category": "Severe Acute Malnutrition (MUAC)",
            "text": "MUAC below 115mm — severe acute malnutrition. Immediate clinical intervention required.",
            "details": [
                "Refer to nearest health facility immediately",
                "Begin ready-to-use therapeutic food (RUTF) if available",
                "Check for medical complications (edema, infections)",
                "Inpatient care may be required",
                "Do NOT delay — this is a life-threatening condition",
            ],
        })
    elif muac_status == "mam":
        recs.append({
            "priority": "urgent",
            "category": "Moderate Acute Malnutrition (MUAC)",
            "text": "MUAC below 125mm — moderate acute malnutrition detected.",
            "details": [
                "Enroll in supplementary feeding program if available",
                "High-energy foods: peanut butter, ghee, oil-enriched porridge",
                "Feed 5-6 small, nutrient-dense meals per day",
                "Clinical follow-up within 2 weeks",
                "Monitor MUAC weekly — refer if below 115mm",
            ],
        })

    # ML-predicted wasting (WHZ-based) — add if not already covered by MUAC
    if wasting and muac_status == "normal":
        recs.append({
            "priority": "urgent",
            "category": "Wasting (Weight-for-Height)",
            "text": "Low weight-for-height detected. Increase calorie and protein intake immediately.",
            "details": [
                "Ready-to-use therapeutic food (RUTF) if available",
                "High-energy foods: peanut butter, ghee, oil-enriched porridge",
                "Feed 5-6 small meals per day",
                "Clinical referral recommended within 48 hours",
            ],
        })

    if stunting:
        recs.append({
            "priority": "important",
            "category": "Stunting",
            "text": "Focus on sustained nutrition improvement for growth recovery.",
            "details": [
                "Protein-rich foods: eggs, milk, dal, fish",
                "Micronutrient-rich foods: green leafy vegetables, fruits",
                "Zinc supplementation may be beneficial",
                "Monitor height-for-age monthly",
            ],
        })

    if anemia:
        recs.append({
            "priority": "important",
            "category": "Anemia",
            "text": "Increase iron intake and absorption.",
            "details": [
                "Iron-rich foods: liver, spinach, lentils, jaggery",
                "Pair with Vitamin C (citrus, tomato) for better absorption",
                "Avoid tea/coffee with meals (inhibits iron absorption)",
                "Consider iron supplementation per clinical guidance",
            ],
        })

    if not recs:
        recs.append({
            "priority": "normal",
            "category": "General",
            "text": "Maintain balanced nutrition.",
            "details": [
                "Continue diverse diet with vegetables, pulses, milk, fruits",
                "Regular growth monitoring every 3 months",
                "Age-appropriate complementary feeding",
            ],
        })

    return recs
