import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_acute():
    X_train, X_test, y_train, y_test = joblib.load("data/processed/acute_data.pkl")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAcute Malnutrition Model Evaluation:")
    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/acute_model.pkl")

    print("Acute model saved.")


if __name__ == "__main__":
    train_acute()
