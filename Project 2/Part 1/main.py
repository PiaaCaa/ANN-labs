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
def RBF(x, n_unit, w,sigma,mu_s, training= False):
    #output hidden layer initilaize, size: N x n_unit
    phi_matrix = np.zeros((x.shape[0], n_unit))

    for i in range(n_unit):
        phi_matrix[:, i] = phi_func(x, sigma=sigma, mu=mu_s[i])

    if training:
        return phi_matrix

    #print("weights: ", w.shape, " values:", w)
    return phi_matrix @ w #np.sum(np.multiply(w, output_h), axis=1) #should be output for learning and sum for other outpur

#batch and least square learning
def least_squares_learning(phi, t):
    return solve(np.transpose(phi) @ phi, np.transpose(phi) @ t)

#return error
def error(f_estimate, f_target):
    return np.sum(np.square(f_estimate-f_target))

def absolute_residual_error(x, y):
    return np.average(np.abs(x-y))

#training: either batch or sequential
def training(x, y, n_units, w_init, sigma, mu_s, batch = True):

    out = RBF(x, n_units, w_init,sigma, mu_s, training=True)

    #batch least squares learningg
    if batch:
        w_update = least_squares_learning(out, y)
        print( "L2 Error:", error(RBF(x, n_units, w_update, training=False, sigma=sigma, mu_s=mu_s), y))
        print("Absolute residual error train:", absolute_residual_error(RBF(x, n_units, w_update, training=False, sigma=sigma, mu_s=mu_s), y))
        return w_update

    #sequential delta rule
    else:
        print('TODO: delta rule')
        epochs = 8

#visualize function prediction
def visualize_results(x, w,y, n_units, sigma, mu_s, title= 'Function prediction', save_as = ""):
    prediction = RBF(x, n_units, w, sigma=sigma, mu_s=mu_s, training=False)

    print("Absolute residual error test:", absolute_residual_error(RBF(x, n_units, w, training=False, sigma=sigma, mu_s=mu_s), y))

    plt.plot(x, prediction, '-o', label = "prediction")
    plt.plot(x, y, label = "target function")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(save_as)
    plt.show()

def plot_RBF_fcts(x, sigma, mu_s):
    for mu in mu_s:
        plt.plot(x, phi_func(x, sigma=sigma, mu = mu))

    plt.show()


def vary_RBFs(x_train, x_test, y_train, y_test, sigmas):
    n_units = np.arange(1, x_train.shape[0]+1)
    print("_____________________________________")
    #print("Sigma:", sigma)

    for sigma in sigmas:
        abs_error = []
        abs_error_train = []
        for n_unit in n_units:
            mu_s = np.linspace(0, 2*np.pi , n_unit)
            w_res = training(x_train, y_train, n_unit, np.random.normal(n_unit), sigma=sigma, mu_s=mu_s, batch=True)

            res = RBF(x_test, n_unit, w_res, training=False, sigma=sigma, mu_s=mu_s)
            print("abs res error test:", absolute_residual_error(res, y_test))
            #abs_error.append(absolute_residual_error(res, y_test))
            abs_error.append(absolute_residual_error(RBF(x_train, n_unit, w_res, training=False, sigma=sigma, mu_s=mu_s), y_train))

        plt.plot(n_units, np.array(abs_error), label = "sigma="+str(sigma))

    plt.plot(n_units, np.ones_like(n_units)*0.001, '--', color ='k', label = "boundary 0.001")
    plt.plot(n_units, np.ones_like(n_units) * 0.01, '--', color ='k', label="boundary 0.01")
    plt.plot(n_units, np.ones_like(n_units) * 0.1, '--', color ='k', label="boundary 0.1")
    plt.title("Absolute residual error compared to number of hidden units")
    plt.legend()
    plt.xlabel("n_units")
    #plt.ylim((0, 0.1))
    plt.ylabel("Absolute residual error")
    plt.savefig("3_1_absolute_residual_error_n_units.pdf")
    plt.show()

if __name__ == '__main__':
    print('Start')

    # train and test inputs
    x_train = np.arange(0, 2 * np.pi + 0.1, 0.1)
    x_test = np.arange(0.05, 2 * np.pi + 0.1, 0.1)

    f1_train = f1(x_train)
    f2_train = f2(x_train)
    f1_test = f1(x_test)
    f2_test = f2(x_test)


    # f1 training
    n_units = 8

    # set RBF values:
    sigma = 1.2
    mus = np.linspace(0, 2 * np.pi, n_units)

    #result
    w_res = training(x_train, f1_train, n_units, np.random.normal(n_units), sigma=sigma, mu_s=mus)

    plot_RBF_fcts(x_train, sigma, mus)

    # visualize_results(x_train, np.random.normal(n_units), f1_train, n_units, title="Function prediction initial values")
    visualize_results(x_test, w_res, f1_test, n_units, sigma, mus, title="Function result on test set", save_as="3.1_result_sin2x.pdf")



    # f2 training

    n_units = x_train.shape[0]
    # set RBF values:
    sigma = 0.05
    mus = np.linspace(0,  2*np.pi, n_units)

    plot_RBF_fcts(x_train, sigma, mus)
    w_res = training(x_train, f2_train, n_units, np.random.normal(n_units), sigma=sigma, mu_s=mus)
    #visualize
    # visualize_results(x_test, np.random.normal(n_units), f2_test, n_units, title="Function prediction initial values")
    visualize_results(x_test, w_res, f2_test, n_units, sigma, mus,  title="Function result on test set", save_as="3.1_result_squaresin2x.pdf")

    #vary the number of hidden nodes
    #vary_RBFs(x_train, x_test, f1_train, f1_test, sigmas=[0.5, 1, 1.2])
    vary_RBFs(x_train, x_test, f2_train, f2_test, sigmas=[0.075,  0.1, 0.5])




