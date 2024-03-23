import numpy as np
import data

def binlogreg_classify(X, w, b):
  '''
  Argumenti
      X:    podatci, np.array NxD
      w Dx1, b skalar: parametri logističke regresije 

  Povratne vrijednosti
      probs: vjerojatnosti razreda c1
   '''
  scores=np.dot(X,w) + b
  probs=  1/(1 + np.exp(-scores))
  return probs
         
        
def binlogreg_train(X,Y_, param_niter, param_delta):
  '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      w np.array Dx1, b skalar: parametri logističke regresije
  '''
  #inicijalizacija w i b
  D=X.shape[1]
  w=np.random.normal(loc=0, scale=1, size=(D, 1))
  b=0

  for i in range(param_niter):
    #klasifikacijske mjere
    scores = np.dot(X, w) + b       #Nx1

    #vjerojatnosti razeda c1
    probs=1/(1 + np.exp(-scores))   #Nx1

    #gubitak
    loss  =  0    # scalar

    #DOVRŠI

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w,b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y =0 # TODO

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)