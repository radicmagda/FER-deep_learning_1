import numpy as np



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

def loss(x, y, w, b):
  """
  args:
     labeled pair x,y and parameters w,b of binary logreg

  retruns:
    loss on example (x,y) given paremeters w and b

  """
  if y==1:
    return

         

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
