
import numpy as np
a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])
c = np.array([11,12,13,14,15])
d = np.array([16,17,18,19,20])

e = np.array([np.array([a,b]).T,np.array([c,d]).T])
print(e)