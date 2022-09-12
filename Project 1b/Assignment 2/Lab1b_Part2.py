# Lab1b PART 2

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Parameters
beta = 0.2
gamma = 0.1
n = 10
tau = 25
x0 = 1.5

# Mackey-Glass time series solved with Euler's method
def mackey_glass(t, time_series):
    xt = time_series[t]

    if t - tau > 0:
        xt_minus_tau = time_series[t-tau]
    else:
        xt_minus_tau = 0

    return xt + beta * xt_minus_tau / (1 + xt_minus_tau ** n) - gamma * xt

#4.1 Data - generate dataset given Mackey-Glass time series
def generate_data(include_noise=False, sigma=0.15):
    t0 = 301
    t1 = 1500
    tf = 2000
    time_lag = 5
    past_values = 4

    time_series = []
    time_series.append(x0)

    # Generate time series from t=0 to t= 2000
    for t in range(0, tf):
        xt_plus_1 = mackey_glass(t, time_series)
        time_series.append(xt_plus_1)

    #Add Gaussian noise if selected
    if include_noise:
        #Set seed for random generation of numbers
        np.random.seed(9)
        #Generate gaussian noise with given variance sigma
        noise = np.random.normal(0, sigma, len(time_series))
        #Add noise
        time_series = time_series + noise


    # Create input and output datasets from t=300 to t=1500
    input_dataset = []
    output_dataset = []
    time_dataset = []
    for t in range(t0, t1+1):
        temp = []
        for l in range(0, (past_values+1)*(time_lag), time_lag):
            temp.append(time_series[t-l])
        input_dataset.append(temp)
        output_dataset.append(time_series[t+time_lag])
        time_dataset.append(t)

    input_array = np.array(input_dataset)
    output_array = np.array(output_dataset)
    time_array = np.array(time_dataset)
    print("Array: ", input_array)
    print("Array: ", output_array)

    ds_size = len(input_array)

    #Splitting input into train, validation and test set
    x_train = input_array[0:800]
    x_val = input_array[800:1000]
    x_test = input_array[1000:]

    #Splitting output into train, validation and test set
    y_train = output_array[0:800]
    y_val = output_array[800:1000]
    y_test = output_array[1000:]

    return x_train, x_val, x_test, y_train, y_val, y_test


def MLP(input_shape, nodes, layers):
    model = Sequential()
    #input layers
    model.add(Dense(5, input_shape = input_shape, activation = "sigmoid"))

    #hidden layers
    for i in range(layers):
        model.add(Dense(nodes[i], activation = "sigmoid"))

    #output layer
    model.add(Dense(1, activation = "relu"))

    return model

def train_model():
    x_train, x_val, x_test, y_train, y_val, y_test = generate_data()
    inputs = 5
    nodes = [3]
    hidden_layers = 1

    model = MLP((inputs,),nodes,hidden_layers)
    model.compile(loss='mean_squared_error')

    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        verbose=1, epochs=100
    )
    plot_loss(history)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 0.4])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MSE]')
  plt.legend()
  plt.grid(True)
  plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
