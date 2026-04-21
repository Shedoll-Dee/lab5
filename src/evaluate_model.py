import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate():
    print("Оценка модели...")

    df = pd.read_csv("data/df_features.csv")

    X = df.drop(columns=["income"])
    y = df["income"]

    model = joblib.load("data/model.pkl")

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)

    print("Accuracy:", accuracy)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == "__main__":
    evaluate()