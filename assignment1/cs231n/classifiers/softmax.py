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
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  for i in xrange(num_train):
    #compute loss
    f = np.dot(X[i], W)
    f -= np.max(f) #shift value of f to prevent numeric instability
    
    exp_f = np.exp(f)
    
    loss += - np.log(exp_f[y[i]] / np.sum(exp_f))
    
    #compute gradient
    for j in xrange(num_classes):
      p = exp_f[j] / np.sum(exp_f)
      dW[:, j] += (p - (j == y[i])) * X[i]
  
  #average
  loss /= num_train
  dW /= num_train
  
  #regularization
  loss += reg * np.sum(W * W)
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  #compute loss
  f = np.dot(X, W)
  f -= np.max(f, axis=1, keepdims=True) #shift value of f to prevent numeric instability
  
  exp_f = np.exp(f)

  loss += np.sum(-np.log(exp_f[np.arange(num_train), y] / np.sum(exp_f, axis=1)))
  
  #compute gradient
  p = exp_f / np.sum(exp_f, axis=1, keepdims=True)
  p[np.arange(num_train), y] -= 1

  dW += np.dot(X.T, p)

  #average
  loss /= num_train
  dW /= num_train
    
  #regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

