import pandas as pd

def load_csv(path="../data/data_LR.csv"):
    """
    Carga el CSV original y devuelve:
    - dataframe completo
    - lista de empresas disponibles
    """
    df = pd.read_csv(path)

    # Normalizar nombres 
    df["Nombre de la Empresa"] = df["Nombre de la Empresa"].str.strip()

    companies = sorted(df["Nombre de la Empresa"].unique())

    return df, companies


def filter_company(df, company_name):
    """Devuelve solo los datos de una empresa."""
    return df[df["Nombre de la Empresa"] == company_name].copy()
