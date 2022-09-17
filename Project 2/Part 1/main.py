import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve


# sin(2x)
def f1(x): return np.sin(2 * x)


# square(2x)
def f2(x):
    return np.where(np.sin(2 * x) >= 0, 1, -1)

#radial basis function
def phi_func(x, sigma=1, mu=0):
    return np.exp(-np.square(x - mu) / (2 * sigma ** 2))

#radial basis network
def RBF(x, n_unit, w, training= False):
    #output hidden layer initilaize, size: N x n_unit
    output_h = np.zeros((x.shape[0], n_units))

    sigma = 1   #same variance for each one
    mu_s = np.linspace(0, 2*np.pi, n_unit) # should pick these ourselves according to our judgement
    for i in range(n_unit):
        output_h[:, i] = phi_func(x, sigma=sigma, mu=mu_s[i])

    if training:
        return output_h

    return np.sum(np.multiply(w, output_h), axis=1) #should be output for learning and sum for other outpur

#batch and least square learning
def least_squares_learning(phi, t):
    return solve(np.transpose(phi) @ phi, np.transpose(phi) @ t)

#return error
def error(f_estimate, f_target):
    return np.sum(np.square(f_estimate-f_target))

def absolute_residual_error(x, y):
    return np.average(np.abs(x-y))

#training: either batch or sequential
def training(x, y, n_units, w_init, batch = True):

    out = RBF(x, n_units, w_init, training=True)

    #batch least squares learningg
    if batch:
        w_update = least_squares_learning(out, y)
        print( "Error:", error(RBF(x, n_units, w_update, training=False), y))
        print("Absolute residual error:", absolute_residual_error(RBF(x, n_units, w_update, training=False), y))
        return w_update

    #sequential delta rule
    else:
        print('TODO: delta rule')
        epochs = 8

#visualize function prediction
def visualize_results(x, w,y, n_units, title= 'Function prediction'):
    prediction = RBF(x, n_units, w, training=False)
    plt.plot(x, prediction, '-o', label = "prediction")
    plt.plot(x, y, label = "target function")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('Start')

    # train and test inputs
    x_train = np.arange(0, 2 * np.pi + 0.1, 0.1)
    x_test = np.arange(0, 2 * np.pi + 0.05, 0.05)

    f1_train = f1(x_train)
    f2_train = f2(x_train)
    f1_test = f1(x_test)
    f2_test = f2(x_test)


    # f1 training
    n_units = 8

    #result
    w_res = training(x_train, f1_train, n_units, w_init=np.random.normal(n_units))

    # visualize_results(x_train, np.random.normal(n_units), f1_train, n_units, title="Function prediction initial values")
    visualize_results(x_train, w_res, f1_train, n_units,  title="Function result")



    # f2 training

    n_units = 5000
    w_res = training(x_train, f2_train, n_units, w_init=np.random.normal(n_units))
    #visualize
    # visualize_results(x_test, np.random.normal(n_units), f2_test, n_units, title="Function prediction initial values")
    visualize_results(x_test, w_res, f2_test, n_units, title="Function result")




