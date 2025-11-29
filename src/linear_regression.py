import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.001, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def predict(self, X):
        return self.w * X + self.b

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def fit(self, X, y):
        n = len(X)

        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            dw = (2/n) * np.sum((y_pred - y) * X)
            db = (2/n) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 500 == 0:
                loss = self.compute_loss(y_pred, y)
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
