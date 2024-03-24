import numpy as np
import data
import binlogreg as binlr

def binlogreg_decfun(w, b):
    return lambda X: binlr.binlogreg_classify(X, w, b)