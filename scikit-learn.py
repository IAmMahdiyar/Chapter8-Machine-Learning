import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

housing = fetch_california_housing()
scaler = StandardScaler()

X_Train = housing.data[250:]

Y_Train = housing.target[250:]

X_Test = housing.data[:250]

Y_Test = housing.target[:250]

print("Fitting California Housing")

cal = MLPRegressor(solver='adam', learning_rate_init=0.001, activation='relu', hidden_layer_sizes=(16, 8), max_iter=5000)
cal.fit(X_Train, Y_Train)

pred = cal.predict(X_Test)

print("California Housing Mean Squad Error: ", np.mean((Y_Test - pred)) ** 2)

X_Train = data[100:]
X_Train = scaler.fit_transform(X_Train)

Y_Train = target[100:]

X_Test = data[:100]
X_Test = scaler.fit_transform(X_Test)

Y_Test = target[:100]

print("Fitting Boston Housing")

bos = MLPRegressor(solver='adam', learning_rate_init=0.001, activation='relu', hidden_layer_sizes=(16, 8), max_iter=5000)
bos.fit(X_Train, Y_Train)

pred = bos.predict(X_Test)

print("Boston Housing Mean Squad Error: ", np.mean((Y_Test - pred)) ** 2)