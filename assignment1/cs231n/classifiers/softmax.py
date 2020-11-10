import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  scores = np.matmul(X, W)
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):

    f = scores[i, :]
    f -= np.max(f) # numerical stablizer 

    score = np.exp(f) # get score, get exp of score
    divider = np.sum(np.exp(f))

    softmax = score/divider # np.exp(f) / np.sum(np.exp(f))
    loss += -np.log(softmax[y[i]]) # L 

    for j in range(num_class):
      dW[:, j] += X[i, :] * softmax[j]
    dW[:, y[i]] -= X[i, :]
      
  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W*W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]

  f = np.matmul(X, W)
  f -= np.max(f, axis=1, keepdims=True) # solve numerical instability 

  softmax = np.exp(f)/np.sum(np.exp(f), axis=1, keepdims=True) # divide np.exp(f) by sum of each column. 
  loss = - (np.sum(np.log(softmax[np.arange(num_train), y]))) / num_train # mean of loss sum
  loss += reg * np.sum(W*W) # reg

  dw_Mat = np.zeros_like(f)
  dw_Mat[np.arange(num_train), y] = 1
  dW = np.matmul(X.T, -dw_Mat + (np.exp(f)/np.sum(np.exp(f), axis=1, keepdims=True))) / num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

