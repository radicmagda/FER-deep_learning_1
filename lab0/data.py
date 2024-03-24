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
   Y_= np.hstack([[Y]*nsamples for Y in Ys]).reshape(nsamples*nclasses,1)
  
   return X,Y_

def eval_perf_binary(Y,Y_):
  """
    calculates classification metrics for binary logistic regression

    Arguments:
    Y -- Y predicted
    Y_ -- Y true

    Returns:
    accuracy, recall, precision
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
    Calculates Average Precision (AP) from ranked

    Arguments:
    ranked_labels -- 

    Returns:
    Average Precision (AP)
    """
  n = ranked_labels.shape[0]
  pos = np.sum(ranked_labels)
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
      

    tp -= x
    fn += x
    fp -= not x
    tn += not x

  return sumprec/pos

def graph_data(X, Y_, Y):
   '''
  X  ... podatci (np.array dimenzija Nx2)
  Y_ ... točni indeksi razreda podataka (Nx1)
  Y  ... predviđeni indeksi razreda podataka (Nx1)
'''

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

def graph_surface(function, rect, offset=0.5, width=256, height=256):
  """Creates a surface plot (visualize with plt.show)

  Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
              ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

  Returns:
    None
  """

  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  #get the values and reshape them
  values=function(grid).reshape((width,height))
  
  # fix the range and offset
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  # draw the surface and the offset
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval)
    
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])

def graph_data(X,Y_, Y, special=[]):
  """Creates a scatter plot (visualize with plt.show)

  Arguments:
      X:       datapoints
      Y_:      groundtruth classification indices
      Y:       predicted class indices
      special: use this to emphasize some points

  Returns:
      None
  """
  # colors of the datapoint markers
  palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
  colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
  for i in range(len(palette)):
    colors[Y_==i] = palette[i]

  # sizes of the datapoint markers
  sizes = np.repeat(20, len(Y_))
  sizes[special] = 40
  
  # draw the correctly classified datapoints
  good = (Y_==Y)
  plt.scatter(X[good,0],X[good,1], c=colors[good], 
              s=sizes[good], marker='o', edgecolors='black')

  # draw the incorrectly classified datapoints
  bad = (Y_!=Y)
  plt.scatter(X[bad,0],X[bad,1], c=colors[bad], 
              s=sizes[bad], marker='s', edgecolors='black')

def class_to_onehot(Y):
  Yoh=np.zeros((len(Y),max(Y)+1))
  Yoh[range(len(Y)),Y] = 1
  return Yoh

if __name__=="__main__":
    np.random.seed(100)
  
    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)
  
    # get the class predictions
    Y = myDummyDecision(X)>0.5
    Y=Y.reshape(Y_.shape)
    print(f"X shape {X.shape}  Y_ shape {Y_.shape}  Y shape {Y.shape} ")


  
    # graph the data points
    graph_data(X, Y_, Y) 
  
    # show the results
    plt.show()