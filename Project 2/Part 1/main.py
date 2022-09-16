import numpy as np
from  matplotlib import pyplot as plt


#sin(2x)
def f1(x): return np.sin(2*x)

#square(2x)
def f2(x):
    return np.where(np.sin(2*x) >= 0, 1, -1)




if __name__ == '__main__':
    print('Start')

    #train and test inputs
    x_train = np.arange(0, 2*np.pi+0.1, 0.1)
    x_test = np.arange(0, 2*np.pi+0.05, 0.05)

    f1_train= f1(x_train)
    f2_train = f2(x_train)
    f1_test = f1(x_test)
    f2_test = f2(x_test)


    # plt.plot(f1_train)
    # plt.plot(f2_train)
    # plt.show()