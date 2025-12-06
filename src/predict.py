import json
import os
import sys
import numpy as np
from linear_regression import LinearRegressionGD
from data_loader import load_dataset, filter_by_company
from train import preprocess

# Ruta absoluta a la carpeta "models"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODELS_DIR = os.path.abspath(MODELS_DIR)

def load_model(company_name: str):
    filename = f"{company_name}_model.json"
    filepath = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(filepath):
        print("Busqué el archivo:", filepath)
        raise FileNotFoundError(f"No existe un modelo para: {company_name}")

    with open(filepath, "r") as f:
        data = json.load(f)

    # Reconstruimos el modelo REAL (usa w y b)
    model = LinearRegressionGD()
    model.w = data["w"]
    model.b = data["b"]

    return model




def predict_next_price(company_name):
    df = load_dataset()
    df_company = filter_by_company(df, company_name)

    if df_company.empty:
        print(f"La empresa '{company_name}' no existe en el dataset.")
        return

    # Procesar datos igual que en train.py
    X, y = preprocess(df_company)

    # Cargar modelo
    model = load_model(company_name)

    # Crear el siguiente punto X_next
    X_next = np.array([[1.1]])  # un poco más allá para simular día siguiente

    prediction = float(model.predict(X_next)[0])

    print(f"\n📈 Predicción para {company_name}:")
    print(f"Precio estimado del próximo día: {prediction:.2f}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py \"NombreEmpresa\"")
        sys.exit(1)

    company = sys.argv[1]
    company = LinearRegressionGD().sanitize_filename(company)

    predict_next_price(company)
