import numpy as np
import tensorflow as tf
import math

a = np.array([[1,2,3,4,5],[6,7,8,9,10],[1,1,1,1,1]])
b = np.array([0,1,2,3,4])
d = np.array([[0],[1],[2],[3],[4]])
e = np.array([[3,3,3], [6,6,6]])
ed = np.array([[3],[3]])

argm = np.array([[1,2,3], [2,4,6]])

# print(np.max(argm[range(2), :], axis=1))

# print("{:.4f}".format(np.pi))

# print(np.max(argm, axis=1).reshape(2,1))

# abc={}
# abc[('1', '2', '3')] = ('a','b','c')
# abc[('1', '2', '4')] = ('a','b','d')
# abc[('1', '1', '2')] = ('a','b','e')
# abc[('4', '4', '1')] = ('a','b','f')

# res = sorted(abc)
# print(res)

# res = sorted(abc, key=lambda x: x[-1])
# print(res)


