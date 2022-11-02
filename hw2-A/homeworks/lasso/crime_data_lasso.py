if __name__ == "__main__":
    from coordinate_descent_algo import train,loss  # type: ignore
else:
    from .coordinate_descent_algo import train,loss

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

import sys
import pandas as pd

"""
1. PolicPerPop numeric
2. NumUnderPov numeric
3. PctLess9thGrade numeric

"""

def MSE(X,y_data,w,b):
    y_model = X@w + b
    mse = np.mean((abs(y_data - y_model))**2)
    return mse

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # generate normal distributed data
    y_train = df_train["ViolentCrimesPerPop"].values #nx1
    y_test = df_test["ViolentCrimesPerPop"].values #nx1
    n = y_train.shape[0]
    n_ = y_test.shape[0]

    d = 5 #number of features selected
    X_train = np.zeros((n,d)) #nxd
    X_test = np.zeros((n_,d)) #nxd
    Features = ["agePct12t29","pctWSocSec","pctUrban","agePct65up","householdsize"]
    for i in range(d):
        X_train[:,i] = df_train[Features[i]].values # ith column
        X_test[:,i] = df_test[Features[i]].values # ith column
    w_start = np.zeros(X_train.shape[1]) #dx1

    # choose the first lambda (the max lambda)
    y_mean = np.mean(y_train)
    l = np.zeros(d)
    for i in range(d):
        l[i] = 2*np.abs(np.dot(X_train[:,i],y_train-y_mean))
    lambda_max = np.max(l)
    
    # training with different lambda values
    trials = 30 # number of trials
    weights = []
    bias = []
    losses = []
    non_zero_w = []
    Lambdas = [lambda_max]
    MSE_train = []
    MSE_test = []
    for i in range(trials):
        if Lambdas[i] > 0.01:
            # train
            _lambda = Lambdas[i]
            w, b = train(X_train, y_train, _lambda, convergence_delta = 1e-4, start_weight = w_start)
            # l = loss(X, y, w, b, _lambda)
            total_nonzeros = np.count_nonzero(w) # total # of nonzeros in estimated weights

            weights.append(w)
            bias.append(b)
            losses.append(l)
            non_zero_w.append( total_nonzeros )
            Lambdas.append(_lambda/2) # reduce by half each time for regularization

            # MSE
            MSE_train.append( MSE(X_train,y_train,w,b) )
            MSE_test.append( MSE(X_test,y_test,w,b) )
        else: #when Lambdas[i] < 0.01:
            break

    # plot 6d
    plt.figure()
    plt.plot(Lambdas[:-1],non_zero_w,'--*')
    plt.xscale('log')
    plt.title(f"HW2 6c")
    plt.xlabel(r"$log(\lambda)$")
    plt.ylabel("# of non-zero features")
    plt.show()

    # plot 6e
    plt.figure()
    plt.plot(Lambdas[:-1],MSE_train,'--*',label = 'training MSE')
    plt.plot(Lambdas[:-1],MSE_test,'--*',label = 'testing MSE')
    plt.xscale('log')
    plt.title(f"HW2 6c")
    plt.xlabel(r"$log(\lambda)$")
    plt.ylabel("# of non-zero features")
    plt.legend()
    plt.show()

    #for lambda = 30:
    w, b = train(X_train, y_train, _lambda = 30, convergence_delta = 1e-4, start_weight = w_start)
    print('features:',w)
    print('the largest coefficient:', np.max(w))
    print('the feature with largest coefficient:', Features[np.argmax(w)]) #pctWSocSec

if __name__ == "__main__":
    main()
