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


def sample_gauss_2d(nclasses, nsamples):
  # create the distributions and groundtruth labels
   """
  stvara C slučajnih bivarijatnih Gaussovih razdioba 
  (prisjetimo se, njih smo u zadatku 1 implementirali razredom Random2DGaussian), 
  te iz svake od njih uzorkuje N podataka. 
  Funkcija treba vratiti 
  matricu X dimenzija (N·C)x2 čiji retci odgovaraju uzorkovanim podatcima te 
  matricu točnih razreda Y dimenzija (N·C)x1 čiji jedini stupac sadrži indeks razdiobe iz koje je uzorkovan odgovarajući podatak. 
  Ako je i-ti redak matrice X uzorkovan iz razdiobe j, onda mora biti Y[i,0]==j.
  """
   Gs=[]
   Ys=[]
   for i in range(nclasses):
     Gs.append(Random2DGaussian())
     Ys.append(i)

  # sample the dataset
   X = np.vstack([G.get_sample(nsamples) for G in Gs])
   Y_= np.hstack([[Y]*nsamples for Y in Ys])
  
   return X,Y_

def eval_perf_binary(Y,Y_):
  """
  Y = Y predicted
  Y_ = Y true
  na temelju predviđenih i točnih indeksa razreda određuje pokazatelje performanse binarne klasifikacije: 
  točnost (engl. accuracy), preciznost (engl. precision) te odziv (engl. recall). 

  """
  tp = sum(np.logical_and(Y==Y_, Y_==True))
  fn = sum(np.logical_and(Y!=Y_, Y_==True))
  tn = sum(np.logical_and(Y==Y_, Y_==False))
  fp = sum(np.logical_and(Y!=Y_, Y_==False))
  recall = tp / (tp + fn)
  precision = tp / (tp + fp)
  accuracy = (tp + tn) / (tp+fn + tn+fp)
  return accuracy, recall, precision

def eval_AP(ranked_labels):
  """
  calculates Average Precision (AP) from ranked labels
  """
  n = len(ranked_labels)
  pos = sum(ranked_labels)
  neg = n - pos
  
  tp = pos
  tn = 0
  fn = 0
  fp = neg
  
  sumprec=0
  #IPython.embed()
  for x in ranked_labels:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)    

    if x:
      sumprec += precision
      
    #print (x, tp,tn,fp,fn, precision, recall, sumprec)
    #IPython.embed()

    tp -= x
    fn += x
    fp -= not x
    tn += not x

  return sumprec/pos