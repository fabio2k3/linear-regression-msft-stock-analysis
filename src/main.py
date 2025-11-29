import pandas as pd
from data_loader import load_dataset, filter_by_company
from train import train_model_for_company

def load_company_list():
    df = load_dataset()
    return sorted(df["Nombre de la Empresa"].unique())

def main():
    print("=== Linear Regression Stock Trainer ===\n")

    companies = load_company_list()

    print("📌 Empresas disponibles:\n")
    for idx, name in enumerate(companies, 1):
        print(f"{idx}. {name}")

    choice = int(input("\nSelecciona una empresa (número): "))
    company = companies[choice - 1]

    print(f"\n🚀 Entrenando modelo para: {company}\n")

    df = load_dataset()
    df_company = filter_by_company(df, company)

    train_model_for_company(df_company, company)

if __name__ == "__main__":
    main()
