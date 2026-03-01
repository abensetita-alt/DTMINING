# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

URL_RED = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
URL_WHITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def load_data(path: str) -> pd.DataFrame:

    return pd.read_csv(path, sep=";")


def load_combined_wine_data(
    url_red: str = URL_RED,
    url_white: str = URL_WHITE
) -> pd.DataFrame:
    """
    Charge red et white, ça ajoute wine_type (0=red, 1=white) et concatène.
    """
    red = load_data(url_red)
    red["wine_type"] = 0

    white = load_data(url_white)
    white["wine_type"] = 1

    df = pd.concat([red, white], axis=0, ignore_index=True)
    return df


def create_quality_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Création de la cible 'quality_grouped' en 3 classes :
      0 : quality <= 4  (Mauvaise)
      1 : 5 <= quality <= 6 (Moyenne)
      2 : quality >= 7 (Bonne)
    """
    df = df.copy()

    def _map_quality(q):
        if q <= 4:
            return 0
        elif q <= 6:
            return 1
        else:
            return 2

    df["quality_grouped"] = df["quality"].apply(_map_quality).astype(int)
    return df


def prepare_data(df: pd.DataFrame):
    """
    Préparation de X et y pour la modélisation.
    - X : toutes les variables explicatives + wine_type
    - y : quality_grouped (0/1/2)
    """
    df = create_quality_grouped(df)

    y = df["quality_grouped"]
    X = df.drop(columns=["quality", "quality_grouped"])

    return X, y


def split_and_scale(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split train / val / test et scaling
    """
    # Split test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Split validation (à partir du train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler