import numpy as np
from linear_regression import LinearRegressionGD


def preprocess(df_company):
    df_company = df_company.sort_values("Fecha")

    # Normalizar X entre 0 y 1
    n = len(df_company)
    X = np.linspace(0, 1, n).reshape(-1, 1)

    y = df_company["Precio de Cierre"].astype(float).values

    return X, y


def train_model_for_company(df_company, company_name):
    X, y = preprocess(df_company)

    model = LinearRegressionGD(lr=0.0001, epochs=3000)
    model.fit(X, y)

    print(f"[OK] Modelo entrenado para {company_name}")
    model.save_model(company_name)

    return model
