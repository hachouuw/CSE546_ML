if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

"""
1. RentMedian numeric
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

    # generate X & w
    X_train = df_train.drop(["ViolentCrimesPerPop"],axis =1).values #nx1
    X_test = df_test.drop(["ViolentCrimesPerPop"],axis=1).values #nx1    
    d = len(df_train.head()) -1 #number of features
    w_start = np.zeros(X_train.shape[1]) #dx1

    # choose the first lambda (the max lambda)
    y_mean = np.mean(y_train)
    l = np.zeros(d)
    for i in range(d):
        l[i] = 2*np.abs(np.dot(X_train[:,i],y_train-y_mean))
    lambda_max = np.max(l)
    print(lambda_max)

    Features = ["agePct12t29","pctWSocSec","pctUrban","agePct65up","householdsize"] # for 6d
    Features_index = [df_train.columns.get_loc(Features[0]) - 1,
                df_train.columns.get_loc(Features[1]) - 1,
                df_train.columns.get_loc(Features[2]) - 1,
                df_train.columns.get_loc(Features[3]) - 1,
                df_train.columns.get_loc(Features[4]) - 1]
   
    # training with different lambda values
    trials = 30 # number of trials
    weights = []
    bias = []
    losses = []
    non_zero_w = []
    Lambdas = [lambda_max]
    MSE_train = []
    MSE_test = []
    Features_weights = []
    for i in range(trials):
        if Lambdas[i] > 0.01:
            # train
            _lambda = Lambdas[i]
            w, b = train(X_train, y_train, _lambda, convergence_delta = 1e-4, start_weight = w_start)
            total_nonzeros = np.count_nonzero(w) # total # of nonzeros in estimated weights

            # for 6d, weight of a particular feature
            Features_weights.append( w[Features_index] )

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
    
    # plot 6c
    plt.figure()
    plt.plot(Lambdas[:-1],non_zero_w,'--*')
    plt.xscale('log')
    plt.title(f"HW2 6c")
    plt.xlabel(r"$log(\lambda)$")
    plt.ylabel("# of non-zero features")
    plt.show()

    # plot 6d
    plt.figure()
    plt.plot(Lambdas[:-1],Features_weights,'--*')
    plt.xscale('log')
    plt.title(f"HW2 6d")
    plt.xlabel(r"$log(\lambda)$")
    plt.ylabel("coefficients")
    plt.legend(Features)
    plt.show()

    # plot 6e
    plt.figure()
    plt.plot(Lambdas[:-1],MSE_train,'--*',label = 'training MSE')
    plt.plot(Lambdas[:-1],MSE_test,'--*',label = 'testing MSE')
    plt.xscale('log')
    plt.title(f"HW 6e")
    plt.xlabel(r"$log(\lambda)$")
    plt.ylabel("# of non-zero features")
    plt.legend()
    plt.show()

    #6f
    #for lambda = 30:
    w, b = train(X_train, y_train, _lambda = 30, convergence_delta = 1e-4, start_weight = w_start)
    print('features:',w)
    print('the largest coefficient:', np.max(w))
    print('the feature with largest coefficient:', Features[np.argmax(w)]) #pctWSocSec
    print('the most negative coefficient:', np.min(w))
    print('the feature with most negative coefficient:', Features[np.argmin(w)]) #pctWSocSec

if __name__ == "__main__":
    main()
