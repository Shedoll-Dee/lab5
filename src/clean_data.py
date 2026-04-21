import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import LabelEncoder


def download_data():
    print("Скачивание датасета...")

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "uciml/adult-census-income",
        path="data/",
        unzip=True
    )

    df = pd.read_csv("data/adult.csv")
    return df


def clean_data():
    df = download_data()

    print("Очистка данных...")

    df = df.drop(['fnlwgt'], axis=1)
    df = df.drop(['education'], axis=1)

    df['income'] = np.where(df['income'] == '<=50K', 0, 1)

    df = df.replace('?', np.nan)
    df = df.dropna()

    cat_columns = [
        'workclass',
        'marital.status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native.country'
    ]

    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df.to_csv("data/df_clean.csv", index=False)

    print("Файл df_clean.csv сохранен")


if __name__ == "__main__":
    clean_data()