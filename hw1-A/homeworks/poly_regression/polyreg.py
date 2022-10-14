"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem

from scipy import stats


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        # raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(x: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given x into an (n, degree) array of polynomial features of degree degree.

        Args:
            x (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        # raise NotImplementedError("Your Code Goes Here")
        x = x.flatten()
        n = len(x)
        d = degree
        X = np.zeros((n,d))
        for i in range(d):
            X[:,i] = x**(i+1)
        return X

    @problem.tag("hw1-A")
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            x (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        # raise NotImplementedError("Your Code Goes Here")

        
        n = len(x)
        d = self.degree
        X = self.polyfeatures(x,d) #nxd matrix

        #data standardization
        # X = stats.zscore(X)
        # X_zscored = np.zeros((n,d))
        # for i in range(d):
        #     X_zscored[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X] #nx(d+1) matrix

        n, d = X_.shape
        # remove 1 for the extra column of ones we added to get the original num features
        d = d - 1

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d + 1)
        # reg_matrix[0, 0] = 0

        # closed form solution, w = (X.TX+lambda I)^-1 X.Ty
        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y) #(d+1)x1


    @problem.tag("hw1-A")
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # raise NotImplementedError("Your Code Goes Here")
        n = len(x)
        d = self.degree
        X = self.polyfeatures(x,d) #nxd matrix

        #data standardization
        # X = stats.zscore(X)
        # X_zscored = np.zeros((n,d))
        # for i in range(d):
        #     X_zscored[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X] #nx(d+1) matrix

        # Y = Xw
        Y = X_@self.weight 

        return Y #nx1


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    # raise NotImplementedError("Your Code Goes Here")
    n = len(a)
    MSE = np.mean((a-b)**2)
    return MSE


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    
    # Fill in errorTrain and errorTest arrays
    # raise NotImplementedError("Your Code Goes Here")
    model = PolynomialRegression(degree, reg_lambda)
    
    for i in range(2,n,1):
        # train model using Xtrain[0:(i+1)]
        model.fit(Xtrain[0:(i+1)], Ytrain[0:(i+1)])

        Y_predict_train = model.predict(Xtrain[0:(i+1)])
        Y_predict_test = model.predict(Xtest)

        # training error
        errorTrain[i] = mean_squared_error(Y_predict_train, Ytrain[0:(i+1)])
        # testing error
        errorTest[i] = mean_squared_error(Y_predict_test, Ytest)

    return errorTrain, errorTest




