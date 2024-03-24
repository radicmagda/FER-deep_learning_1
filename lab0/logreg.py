import numpy as np
import data
import binlogreg as binlr

def binlogreg_decfun(w, b):
    return lambda X: binlr.binlogreg_classify(X, w, b)

def logreg_train(X, Y_):
    """
    Trains a multi-class logistic regression model

    Args:
        X: ndarray NxD input data matrix,
        Y_: ndarray Nx1 target variable,

    Returns:
        W: ndarray CxD weight matrix,
        b: ndarray Cx1 bias vector,

    Note:
        C = number of classes = MAX(Y_) + 1.
    """
