import numpy as np

X = [[1,2,3],[2,3,4]]
Y = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]


print(np.array(Y[:])-np.array(X[0]))
gap = np.array(Y[:])-np.array(X[0])
print(np.sqrt(gap**2))
L2 = np.sqrt(gap**2)