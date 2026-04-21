import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


def train():
    print("Обучение модели...")

    df = pd.read_csv("data/df_features.csv")

    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = SGDClassifier(random_state=42)

    model.fit(X_train, y_train)

    joblib.dump(model, "data/model.pkl")

    print("Модель сохранена: model.pkl")


if __name__ == "__main__":
    train()