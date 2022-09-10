# Lab1b PART 2

import matplotlib.pyplot as plt

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
def generate_data():
    t0 = 301
    t1 = 1500
    tf = 2000
    time_lag = 5
    past_values = 4

    time_series = []
    time_series.append(x0)

    for t in range(0, t1+1):
        xt_plus_1 = mackey_glass(t, time_series)
        time_series.append(xt_plus_1)

    time_series = time_series[t0:t1+1]

    final_dataset = []
    for t in range(t0, t1+1):
        temp = []
        for l in range(0, (past_values+1)*(time_lag), time_lag):
            temp.append(time_series[t-l])
        final_dataset.append(temp)

    #print(final_dataset)
    plt.plot(time_series)
    print(len(time_series))
    plt.xticks(range(1200), range(301,1501))
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
