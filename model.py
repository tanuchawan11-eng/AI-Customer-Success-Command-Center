import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_churn_model(csv_path="customer_churn_sample.csv"):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Define features and target
    features = [
        "tickets_last_30_days",
        "sla_violations",
        "reopened_tickets",
        "avg_response_time",
        "sentiment_score",
        "usage_drop_percent",
    ]
    target = "churned"

    X = df[features]
    y = df[target]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save model
    joblib.dump(model, "churn_model.pkl")

    return accuracy


if __name__ == "__main__":
    accuracy = train_churn_model()
    print(f"Model trained! Accuracy: {accuracy:.2f}")
