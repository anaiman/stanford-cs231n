import numpy as np
from random import shuffle

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
  input_dimension = W.shape[0] # D dimension
  num_classes = W.shape[1] # C classes
  num_train = X.shape[0] # N examples in minibatch
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)  # Shift values to avoid numerical instability
    
    correct_class_score = scores[y[i]]
    exp_sum = np.sum(np.exp(scores))
    loss += -correct_class_score + np.log(exp_sum)
    
    for j in xrange(num_classes):
      p = np.exp(scores[j]) / exp_sum
      dW[:, j] += p * X[i]
    dW[:, y[i]] -= X[i]
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1] # C classes
  num_train = X.shape[0] # N examples in minibatch
  
  scores = X.dot(W)
  scores -= np.max(scores, axis=1).reshape(-1,1)
  correct_class_scores = scores[np.arange(num_train), y].reshape(-1,1)
  
  exp_scores = np.exp(scores)
  exp_sum = np.sum(exp_scores, axis=1).reshape(-1,1)
  loss = np.mean(-correct_class_scores + np.log(exp_sum))
  
  p = exp_scores / exp_sum
  p_mask = np.zeros_like(p)
  p_mask[np.arange(num_train), y] = 1
  p = p - p_mask
  dW = np.transpose(X).dot(p)
  #for i in xrange(num_train):
  #  dW[:, y[i]] -= X[i]
  dW /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

