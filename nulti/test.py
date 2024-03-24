import numpy as np
import matplotlib.pyplot as plt 
import numpy as np 
import math 
import data
  

probs=np.array([0.3,0.2,0.4,0.6,0.5,0.7]).reshape(6,1)
labels=np.array([0,0,1,1,0,1]).reshape(6,1)
gs=probs-labels
print(gs)
print(gs.shape)
x=np.random.rand(6,3)
print(x)
print(x.shape)
y=np.dot(gs.T, x)*60
print(y)
print(y.shape)