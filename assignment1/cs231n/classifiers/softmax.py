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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = np.exp(X[i].dot(W))
    #Deep learning book: softmax can overflow
    sum_scores = np.sum(scores)
    correct_class_score = scores[y[i]]/sum_scores
    loss += -np.log(correct_class_score)
    for j in range(num_classes):
      if j != y[i]:
        dW[:, j] += (scores[j]/sum_scores) * X[i]
      else:
        dW[:, j] += (scores[j]/sum_scores - 1) * X[i]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.exp(X.dot(W))
  sum_scores = np.sum(scores, axis=1)
  correct_class_scores = scores[np.arange(num_train), y]
  normalized_scores = scores/sum_scores[:, np.newaxis]
  loss_all = - np.log(correct_class_scores/sum_scores)
  loss = np.mean(loss_all) + 0.5 * reg * np.sum(W * W)

  normalized_scores[np.arange(num_train), y] = - (sum_scores - correct_class_scores)/sum_scores
  dW = X.T.dot(normalized_scores)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

