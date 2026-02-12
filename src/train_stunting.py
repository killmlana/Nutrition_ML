import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_stunting():
    X_train, X_test, y_train, y_test = joblib.load("data/processed/stunting_data.pkl")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nStunting Model Evaluation:")
    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/stunting_model.pkl")

    print("Stunting model saved.")


if __name__ == "__main__":
    train_stunting()
