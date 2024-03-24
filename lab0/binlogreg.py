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
  N=X.shape[0]
  D=X.shape[1]
  w=np.random.normal(loc=0, scale=1, size=(D, 1))  #Dx1
  b=0

  for i in range(param_niter):
    #klasifikacijske mjere
    scores = np.dot(X, w) + b       #Nx1

    #vjerojatnosti razeda c1
    probs=1/(1 + np.exp(-scores))   #Nx1

    #gubitak
    losses = -np.where(Y_ == 1, np.log(probs), np.log(1 - probs))
    loss = np.mean(losses)   # scalar
    
    # dijagnostički ispis
    if i % 100 == 0:
      print("iteration {}: loss {}".format(i, loss))

    # derivacije gubitka po klasifikacijskim mjerama tj vektor gs
    dL_dscores = probs-Y_    # N x 1
    
    # gradijenti parametara
    grad_w =np.dot(dL_dscores.T, X)*1/N   # D x 1
    grad_w=grad_w.T
    grad_b = np.mean(dL_dscores)     # 1 x 1

    # poboljšani parametri
    w += -param_delta * grad_w
    b += -param_delta * grad_b

  return w, b

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(2, 100)
    print(f"Y_ shape: {Y_.shape}")

    # train the model
    w,b = binlogreg_train(X, Y_, param_niter=3000,  param_delta=0.1)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y =np.where(probs >= 0.5, 1, 0) #

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    Y_ranked = Y_[probs.argsort(axis=0)[:, ::-1]].reshape(Y_.shape)
    AP = data.eval_AP(Y_ranked)
    print (accuracy, recall, precision, AP)