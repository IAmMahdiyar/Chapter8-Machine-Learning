import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing

tf.random.set_seed(42)
housing = fetch_california_housing()

X_Train = housing.data[250:]

Y_Train = housing.target[250:]

X_Test = housing.data[:250]

Y_Test = housing.target[:250]

model = keras.Sequential([
    keras.layers.Dense(activation='relu', units=20),
    keras.layers.Dense(activation='relu', units=8),
    keras.layers.Dense(units=1)
])

print("California Housing Compiling")
model.compile(optimizer=tf.optimizers.Adam(0.02), loss='mean_squared_error')

print("California Housing Fitting")
model.fit(X_Train, Y_Train, epochs=200)
