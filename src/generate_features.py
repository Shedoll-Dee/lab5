import pandas as pd


def generate_features():
    print("Генерация новых признаков...")

    df = pd.read_csv("data/df_clean.csv")

    # новый признак
    df["capital_total"] = df["capital.gain"] - df["capital.loss"]

    # ещё один
    df["hours_age_ratio"] = df["hours.per.week"] / df["age"]

    df.to_csv("data/df_features.csv", index=False)

    print("Файл df_features.csv сохранен")


if __name__ == "__main__":
    generate_features()