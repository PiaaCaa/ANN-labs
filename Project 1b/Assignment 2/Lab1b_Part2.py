# Lab1b PART 2

from re import X
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tabulate import tabulate

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


def test_fct(beta = 0.2, gamma=0.1, n=10, tau= 25, x0 = 1.5):
    ts = np.arange(301, 1501).transpose()

    #TODO noise
    def x_euler(max_t):
        #create array for result
        y = np.zeros(max_t+1)
        y[0] = x0

        for t in range(0, max_t):
            index_25 = t - 25
            if index_25 < 0:
                res_25 = 0
            else:
                res_25 = y[index_25]

            y[t+1] = y[t] + beta*res_25/(1+res_25**10) - gamma*y[t]
        return y


    y_res = x_euler(1500+5)
    
    input = np.array([y_res[ts-20], y_res[ts-15], y_res[ts-10], y_res[ts - 5], y_res[ts]])
    output = y_res[ts+5].transpose()
    print(input.shape, output.shape)
    return input, output


def generate_data_2(include_noise=False, sigma=0.15):

    #get input and output values
    input_array, output_array = test_fct()

    input_array = input_array.transpose () # now 1200 x 5
    output_array = output_array.transpose()

    #Splitting input into train, validation and test set
    x_train = input_array[ 0:800]
    x_val = input_array[800:1000]
    x_test = input_array[1000:]

    #Splitting output into train, validation and test set
    y_train = output_array[0:800]
    y_val = output_array[800:1000]
    y_test = output_array[1000:]

    return x_train, x_val, x_test, y_train, y_val, y_test


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


def MLP(input_shape, nodes, reg_lamda =  0):
    
    initializer = tf.keras.initializers.GlorotNormal(seed = 3)

    model = Sequential()
    #input layers
    model.add(Dense(5, input_shape = input_shape, activation = "sigmoid", kernel_initializer=initializer))

    #hidden layers
    for n_nodes in nodes:
        model.add(Dense(n_nodes, activation = "sigmoid", kernel_initializer=initializer, kernel_regularizer=keras.regularizers.l2(reg_lamda), bias_regularizer=keras.regularizers.l2(reg_lamda)))

    #output layer
    model.add(Dense(1, kernel_initializer=initializer, activation = 'linear'))

    return model


#train the MLP network
def train_model(nodes, add_noise = False, sigma = 0.15, reg_lamda= 0):
    x_train, x_val, x_test, y_train, y_val, y_test = generate_data(include_noise=add_noise, sigma=sigma)
    inputs = 5

    #create model
    model = MLP((inputs,),nodes, reg_lamda =reg_lamda )

    #early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta = 0.00002,
    verbose=1,
    mode='min',
    patience = 200,
    restore_best_weights=True
)


    #compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                optimizer = tf.keras.optimizers.SGD(learning_rate = 0.15) ,
                metrics = ['mse']) 

    #model.summary()
    history = model.fit(
        x_train,
        y_train,
        batch_size = x_train.shape[0] , 
        validation_data = (x_val, y_val),
        verbose=0, epochs=15000, 
        callbacks = [early_stopping]
    )
    plot_loss(history)

    results_test = model.evaluate(x_test, y_test)
    print("Test results: ", results)

    ts = np.arange(301, 1501)
    res_test = model.predict(x_test)
    res_val = model.predict(x_val)
    res_train = model.predict(x_train)
    
    
    plt.plot( ts[1000:], res_test,  label = "predicted test")
    plt.plot(ts[1000:], y_test,  label = "target test")
    #plt.plot(ts[0:800], y_train, label="predicted train")
    #plt.plot(ts[0:800], res_train,  label = "gt train")
    #plt.plot(ts[800:1000], y_val, label="predicted train")
    #plt.plot(ts[800:1000], res_val,  label = "gt train")
    plt.legend()
    plt.savefig("3_2_2_prediction.png")
    plt.show()
    return np.min(history.history["val_mse"]), results_test,  np.min(history.history["loss"])


    

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MSE]')
  plt.legend()
  plt.grid(True)
  plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #do grid search
    nodes_combinations = [[5, 6]] #[[5, 3], [5, 6], [5, 9] ]


    #todo get nodes of combinations
    regularisation = [0]#[0, 0.3, 0.6]
    noise = [True]#[False, True, True]
    sigmas = [0.15]#[0, 0.05, 0.15]


    model_performances = []
    results = []
    res_test_train_val =[]
    for reg in regularisation:
        
        res_tabular_val = []
        res_tabular_train = []
        res_tabular_test = []
        for nodes in nodes_combinations:
            res_val_sigma = [str(nodes[0])+"x"+str(nodes[1])]
            res_train_sigma = [str(nodes[0])+"x"+str(nodes[1])]
            res_test_sigma = [str(nodes[0])+"x"+str(nodes[1])]

            for add_noise, sigma in zip(noise, sigmas):
                res_val, res_test, res_train = train_model(nodes, add_noise = add_noise, sigma=sigma, reg_lamda = reg) 
                model_performances.append(['noise: '+ str(sigma)+ ' nodes'+str(nodes[0])+"x"+str(nodes[1])+ " reg:"+ str(reg)+": ", res_val])
                #results.append((add_noise, sigma, reg, nodes, res_val, res_test, res_train))
                results.append([add_noise, sigma, reg, nodes, res_val])
                res_test_train_val.append([add_noise, sigma, reg, nodes, res_val, res_test, res_train])

                res_val_sigma.append(res_val)
                res_test_sigma.append(res_test)
                res_train_sigma.append(res_train)

            res_tabular_val.append(res_val_sigma)
            res_tabular_train.append(res_train_sigma)
            res_tabular_test.append(res_test_sigma)

        print("Regularization:", reg)
        print("VALIDATION:")
        print(tabulate(res_tabular_val, headers=["architecture", "No noise", "s=0.05", "s=0.15"]))

        print("TEST")
        print(tabulate(res_tabular_test, headers=["architecture", "No noise", "s=0.05", "s=0.15"]))

        print("TRAIN")
        print(tabulate(res_tabular_train, headers=["architecture", "No noise", "s=0.05", "s=0.15"]))
    #print(model_performances)
    #print(results)

    #d = results
    #print(tabulate)
