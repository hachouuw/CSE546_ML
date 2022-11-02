from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    A = 2*(np.linalg.norm(X, axis = 0)**2) #2-norm square of each column
    return A


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    n,d = X.shape
    w = weight
    b = (1/n)*(np.sum(y - X@w)) #np.mean(y - X@w) # calculate bias
    c = np.zeros(d) #scalar
    for k in range(d):
        w_ = np.copy(w)
        w_[k] = 0
        ybWX = y - (b + X@w_)
        c[k] = 2* np.dot(X[:,k],ybWX) #inner product
        
        if c[k] < -_lambda:
            w[k] = (c[k] + _lambda)/a[k]
        elif c[k] > _lambda:
            w[k] = (c[k] - _lambda)/a[k]
        else:
            w[k] = 0
    return w,b


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    n,d = X.shape
    w = weight  
    b = bias
    loss = np.zeros(n)
    for i in range(n):  
        loss[i] = (np.dot(X[i,:],w) + b - y[i])**2
    Loss = np.sum(loss) + _lambda*(np.linalg.norm(w,ord = 1))

    return Loss

@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    # old_w: Optional[np.ndarray] = None

    CONVERGED = False
    old_w = start_weight
    while CONVERGED is False:
        w,b = step(X, y, old_w, a, _lambda)
        old_w = np.copy(w)    
        CONVERGED = convergence_criterion(w, old_w, convergence_delta)
    return w,b


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    max_change = np.max(np.abs(weight - old_w))
    if max_change < convergence_delta:
        CONVERGED = True
    else:
        CONVERGED = False
    return CONVERGED


@problem.tag("hw2-A")
def main():
    """
    repeatedly call `train` and record various quantities around non zero entries in returned weight vector.
    Use all of the functions above to make plots.
    """
    n,d,k,sigma = 500,1000,100,1

    # generate normal distributed data
    X_ = np.random.normal(0.5, sigma, (n,d)) #generate my own X (nxd) with random distribution
    w_data = np.zeros(d) # given w to generate y
    for j in range(k):
        w_data[j] = j/k

    # standardize X (z-scored)
    mean = np.mean(X_,axis=0)
    std = np.std(X_,axis=0)
    X = np.zeros((n,d))
    for i in range(d):
        X[:,i] = (X_[:,i] - mean[i])/std[i]

    noise = np.random.normal(0, sigma, 500) # generate noise, bias = 0
    y = X@w_data + noise # generate y = wx + noise

    # choose the first lambda (the max lambda)
    y_mean = np.mean(y)
    l = np.zeros(d)
    for i in range(d):
        l[i] = 2*abs(np.dot(X[:,i],y-y_mean))
    lambda_max = np.max(l)

    # training with different lambda values
    trials = 15 # number of trials
    weights = []
    bias = []
    non_zero_w = []
    Lambdas = [lambda_max]
    FDR = []
    TPR = []
    for i in range(trials):
        _lambda = Lambdas[i]
        w, b = train(X, y, _lambda, convergence_delta = 1e-4, start_weight = None)
        weights.append(w)
        bias.append(b)
        total_nonzeros = np.count_nonzero(w) # total # of nonzeros in estimated weights
        non_zero_w.append( total_nonzeros )
        Lambdas.append(_lambda/2) # reduce by half each time for regularization

        correct_nonzero = 0
        incorrect_nonzero = 0
        for j in range(d):
            if w_data[j] == 0:
                if w[j] == 0:
                    correct_nonzero += 1 
                else: 
                    incorrect_nonzero += 1

        if total_nonzeros == 0: #divide by zero
            FDR.append( 0 ) # false discovery rate
            TPR.append( 0 ) #true positive rate 
        else:
            FDR.append( incorrect_nonzero/total_nonzeros ) # false discovery rate
            TPR.append( correct_nonzero/k ) #true positive rate 

    # plot 5a
    plt.figure()
    plt.plot(Lambdas[:-1],non_zero_w)
    plt.xscale('log')
    plt.title(f"HW2 5a")
    plt.xlabel(r"$log(\lambda)$")
    plt.ylabel("# of non-zero features")
    plt.show()

    # plot 5b
    plt.figure()
    plt.plot(FDR,TPR,'*')
    plt.xscale('log')
    plt.title(f"HW2 5b")
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.show()

    # return the lambda that gives us the ideal situation
    print(Lambdas[np.argmax(TPR)])


if __name__ == "__main__":
    main()
