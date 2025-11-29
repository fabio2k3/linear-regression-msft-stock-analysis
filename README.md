# Linear Regression on Microsoft Stock Data

A practical machine learning project implementing **linear regression from scratch** to model and predict Microsoft's monthly closing stock prices using **gradient descent** and **mean squared error**.

---

## 📌 Project Overview

This project applies the fundamental concepts of **Supervised Learning** and **Regression Models**, focusing on:

* Linear regression model implementation
* Manual computation of predictions
* Mean Squared Error (MSE) as loss function
* Gradient Descent optimization
* Hyperparameter tuning (learning rate, epochs)
* Data preprocessing and visualization

The project uses monthly closing price data for **Microsoft (MSFT)** from 2018 to 2024.

---

## 📁 Dataset

The dataset contains the following columns:

* **Company Name**
* **Date (monthly)**
* **Closing Price**

For modeling, the date column is transformed into a numerical index (MonthIndex) representing the time progression.

---

## 🧹 Data Preparation

1. Load CSV file
2. Sort by date
3. Create a new feature:

   * `MonthIndex = 0, 1, 2, ...` representing each month
4. Select input `X = MonthIndex` and target `y = Closing Price`
5. Optionally normalize data for better gradient descent convergence

---

## 🧮 Linear Regression Model

The hypothesis function:

```
ŷ = w * x + b
```

Where:

* `w` = weight
* `b` = bias/intercept

---

## 🎯 Loss Function: Mean Squared Error

```
MSE = (1/n) * Σ(yi - ŷi)^2
```

This measures the average squared error between predictions and actual prices.

---

## 🔁 Gradient Descent Optimization

The partial derivatives of the loss with respect to the parameters are:

```
dw = -(2/n) * Σ[xi * (yi - ŷi)]
db = -(2/n) * Σ[(yi - ŷi)]
```

Parameter updates:

```
w = w - learning_rate * dw
b = b - learning_rate * db
```

Hyperparameters:

* Learning rate (α)
* Number of epochs/iterations

---

## 📈 Visualizations

1. **Loss vs Epochs Curve**

   * Shows whether gradient descent is converging
2. **Regression Line vs Real Data**

   * Scatter plot of actual prices
   * Regression line predicted by the model

---

## 📊 Model Evaluation

You can compute:

* MSE
* MAE (Mean Absolute Error)
* RMSE (Root Mean Square Error)

And interpret `w` to understand the price trend:

* If `w` > 0, price increases over time
* Magnitude of `w` indicates the monthly change

---

## 🔮 Future Predictions

To predict for a future date:

* Convert the date to its corresponding MonthIndex
* Apply:

```
ŷ = w * MonthIndex + b
```

---

---

## 🚀 How to Run

1. Install dependencies:

```
pip install numpy pandas matplotlib
```

2. Run the notebook or main script
3. Visualize results and adjust learning rate/epochs if needed

---

## 🎓 Learning Objectives

By completing this project you will understand:

* How linear models learn
* How loss functions measure performance
* How gradient descent adjusts parameters
* The effect of learning rate and number of epochs
* How to prepare real financial data for ML

---

## 📬 Contact

Feel free to open issues or pull requests if you'd like to extend the project.
