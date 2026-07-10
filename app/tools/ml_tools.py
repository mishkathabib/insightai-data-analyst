import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from app.utils.preprocessing import preprocess_data


def train_classification_model(df: pd.DataFrame, target: str):
    if target not in df.columns:
        return f"{target} is not a column in the dataset."

    X, y = preprocess_data(df, target)

    if len(X) < 10:
        return "Not enough rows to train a model."

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results[name] = round(accuracy, 3)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    if hasattr(best_model, "feature_importances_"):
        feature_importance = pd.Series(
            best_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False).head(10)
    else:
        feature_importance = pd.Series(
            abs(best_model.coef_[0]),
            index=X.columns
        ).sort_values(ascending=False).head(10)

    return {
        "model_results": results,
        "best_model": best_model_name,
        "best_accuracy": results[best_model_name],
        "top_features": feature_importance
    }