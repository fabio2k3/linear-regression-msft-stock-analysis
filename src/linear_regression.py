import numpy as np
import json
import os
import re

class LinearRegressionGD:
    def __init__(self, lr=0.0001, epochs=3000):
        self.lr = lr
        self.epochs = epochs
        self.w = float(np.random.randn())
        self.b = float(np.random.randn())

    def predict(self, X):
        return (self.w * X) + self.b

    def mse(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def fit(self, X, y):
        X = X.astype(float).ravel()
        y = y.astype(float).ravel()

        n = len(X)

        for epoch in range(1, self.epochs + 1):
            y_pred = self.predict(X)

            dw = (2/n) * np.sum((y_pred - y) * X)
            db = (2/n) * np.sum(y_pred - y)

            self.w = float(self.w - self.lr * dw)
            self.b = float(self.b - self.lr * db)

            if epoch % 500 == 0:
                loss = self.mse(y_pred, y)
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def sanitize_filename(self, name):
        """Elimina caracteres ilegales para Windows."""
        return re.sub(r'[\\/*?:"<>|]', "_", name)

    def save_model(self, company_name):
        company_name_sanitized = self.sanitize_filename(company_name)
        model_data = {
            "w": self.w,
            "b": self.b,
            "company": company_name
        }

        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{company_name_sanitized}_model.json")

        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=4)

        print(f"[OK] Modelo guardado en {model_path}")
