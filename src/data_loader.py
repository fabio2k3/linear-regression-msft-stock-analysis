import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "data_LR.csv")


def load_dataset():
    df = pd.read_csv(CSV_PATH)

    # Limpieza
    df["Precio de Cierre"] = (
        df["Precio de Cierre"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["Fecha"] = pd.to_datetime(df["Fecha"])

    return df


def filter_by_company(df, company_name):
    return df[df["Nombre de la Empresa"] == company_name].copy()
