from matplotlib import pyplot as plt
import numpy as np

class Random2DGaussian:
  """Random bivariate normal distribution sampler

  Hardwired parameters:
      d0min,d0max: horizontal range for the mean
      d1min,d1max: vertical range for the mean
      scalecov: controls the covariance range 

  Methods:
      __init__: creates a new distribution

      get_sample(n): samples n datapoints

  """

  d0min=0 
  d0max=10
  d1min=0 
  d1max=10
  scalecov=5
  
  def __init__(self):
    dw0,dw1 = self.d0max-self.d0min, self.d1max-self.d1min
    mean = (self.d0min,self.d1min)
    mean += np.random.random_sample(2)*(dw0, dw1)
    eigvals = np.random.random_sample(2)
    eigvals *= (dw0/self.scalecov, dw1/self.scalecov)
    eigvals **= 2
    theta = np.random.random_sample()*np.pi*2
    R = [[np.cos(theta), -np.sin(theta)], 
         [np.sin(theta), np.cos(theta)]]
    Sigma = np.dot(np.dot(np.transpose(R), np.diag(eigvals)), R)
    self.get_sample = lambda n: np.random.multivariate_normal(mean,Sigma,n)

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

if __name__=="__main__":
    np.random.seed(100)
    G=Random2DGaussian()
    X=G.get_sample(100)
    plt.scatter(X[:,0], X[:,1])
    plt.show()