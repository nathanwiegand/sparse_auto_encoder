import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import utils

# This method is used to enforce the dimensionality of matrices since NumPy is a
# bit aggressive about allowing operators over non-matching dimensions.
def ASSERT_SIZE(matrix, shape):
  if matrix.shape != shape:
    raise AssertionError("Wrong shape: %s expexted: %s" %
                            (matrix.shape, shape))

# This wraps the parameters for the sparse autoencoder.
class SparseAutoEncoderOptions:
  #  These network parameters are specified by by Andrew Ng specifically for
  #  the MNIST data set here:
  #     [[http://ufldl.stanford.edu/wiki/index.php/Exercise:Vectorization]]
  def __init__(self,
               visible_size,
               hidden_size,
               sparsity = 0.1,
               learning_rate = 3e-3,
               beta = 3,
               output_dir = "output",
               max_iterations = 500):
    self.visible_size = visible_size
    self.hidden_size = hidden_size
    self.sparsity_param = sparsity
    self.learning_rate = learning_rate
    self.beta = beta
    self.output_dir = output_dir
    self.max_iterations = max_iterations

class SparseAutoEncoderSolution:
  def __init__(self, W1, W2, b1, b2):
    self.W1 = W1
    self.W2 = W2
    self.b1 = b1
    self.b2 = b2

# The SparseAutoEncoder object wraps all the data needed in order to train a
# sparse autoencoder.  Its constructor takes a SparseAutoEncoderOptions and a
# v x m matrix where v is the size of the visible layer of the network.
class SparseAutoEncoder:
  def __init__(self, options, data):
    self.options = options
    self.data = data

    self.frame_number = 0

  # Convert the matrices to a flat vector.  This is needed by 'fmin_l_bfgs_b'.
  def flatten(self, W1, W2, b1, b2):
    return np.array(np.hstack([W1.ravel('F'), W2.ravel('F'),
                               b1.ravel('F'), b2.ravel('F')]), order='F')

  # Convert the flat vector back to the W1, W2, b1, and b2 matrices.
  def unflatten(self, theta):
    hidden_size = self.options.hidden_size
    visible_size = self.options.visible_size
    hv = hidden_size * visible_size
    W1 = theta[0:hv].reshape([hidden_size, visible_size], order='F')
    W2 = theta[hv:2*hv].reshape([visible_size, hidden_size], order='F')
    b1 = theta[2*hv:2*hv+hidden_size].reshape([hidden_size, 1], order='F')
    b2 = theta[2*hv+hidden_size:].reshape([visible_size, 1], order='F')
    return (W1, W2, b1, b2)

  # Create the random values for the parameters to begin learning.
  def initialize_parameters(self):
    hidden_size = self.options.hidden_size
    visible_size = self.options.visible_size
    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random([hidden_size, visible_size]) * 2 * r - r;
    W2 = np.random.random([visible_size, hidden_size]) * 2 * r - r;
    b1 = np.zeros([hidden_size, 1])
    b2 = np.zeros([visible_size, 1])

    return self.flatten(W1, W2, b1, b2)

  # <div class='math'>1/(1 + e^{-x})</div>
  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  # ==Forward pass==
  # Note: even though the dimensionality doesn't match because <p>$$b1$$</p>
  # is a vector, numpy will apply b1 to every column.
  def feed_forward(self, x, W1, W2, b1, b2):
    visible_size = self.options.visible_size
    hidden_size = self.options.hidden_size
    ASSERT_SIZE(W1, (hidden_size, visible_size))

    m = x.shape[1]
    z2 = np.dot(W1, x) + b1
    a2 = self.sigmoid(z2)
    ASSERT_SIZE(a2, (hidden_size, m))

    z3 = np.dot(W2, a2) + b2 # W2 * a2 + b2
    a3 = self.sigmoid(z3)
    ASSERT_SIZE(a3, (visible_size, m))
    return a2, a3

  # Compute the cost function J and the gradient for an input.  Note that this
  # takes a flattened W1, W2, b1, b2 because of fmin_l_bfgs_b.
  def sparse_autoencoder(self, theta):
    visible_size = self.options.visible_size
    hidden_size = self.options.hidden_size
    lamb = self.options.learning_rate
    rho = self.options.sparsity_param
    beta = self.options.beta

    x = self.data
    m = x.shape[1]

    W1, W2, b1, b2 = self.unflatten(theta)
    ASSERT_SIZE(W1, (hidden_size, visible_size))
    ASSERT_SIZE(W2, (visible_size, hidden_size))
    ASSERT_SIZE(b1, (hidden_size, 1))
    ASSERT_SIZE(b2, (visible_size, 1))

    if self.frame_number % 100 == 0:
      utils.save_as_figure(W1.T,
                           "%s/w1frame%03d.png" % (self.options.output_dir,
                                                   self.frame_number))
      utils.save_as_figure(W2.T,
                           "%s/w2frame%03d.png" % (self.options.output_dir,
                                                   self.frame_number))
    self.frame_number += 1

    a2, a3 = self.feed_forward(x, W1, W2, b1, b2)

    # Compute average activation for an edge over all data
    rho_hat = np.mean(a2, 1)[:, np.newaxis]
    ASSERT_SIZE(rho_hat, (hidden_size, 1))
    kl = rho*np.log(rho/rho_hat) + (1-rho)*np.log((1-rho)/(1-rho_hat))

    cost = 0.5/m * np.sum((a3 - x)**2) + \
           (lamb/2.)*(np.sum(W1**2) + np.sum(W2**2)) + \
           beta*np.sum(kl)

    # We set <span class='math'>y</span> equal to the input since we're learning
    # an identity function
    y = x
    delta3 = -(y - a3) * a3*(1-a3)
    ASSERT_SIZE(delta3, (visible_size, m))

    sparsity = -rho/rho_hat + (1-rho)/(1-rho_hat)
    ASSERT_SIZE(sparsity, (hidden_size, 1))

    delta2 = (np.dot(W2.T, delta3) + beta * sparsity) * a2 * (1-a2)
    ASSERT_SIZE(delta2, (hidden_size, m))

    W2_grad = 1./m * np.dot(delta3, a2.T) + lamb * W2
    ASSERT_SIZE(W2_grad, (visible_size, hidden_size))

    # [:, newaxis] makes this into a matrix
    b2_grad = 1./m * np.sum(delta3, 1)[:, np.newaxis]
    ASSERT_SIZE(b2_grad, (visible_size, 1))

    # sum the rows of delta3 and then mult by  1/m
    W1_grad = 1./m * np.dot(delta2, x.T) + lamb * W1
    ASSERT_SIZE(W1_grad, (hidden_size, visible_size))

    b1_grad = 1./m * np.sum(delta2, 1)[:, np.newaxis]
    ASSERT_SIZE(b1_grad, (hidden_size, 1))

    grad = self.flatten(W1_grad, W2_grad, b1_grad, b2_grad)
    return (cost, grad)

  # Actually run gradient descent.  Call mySparseAutoEncoder.learn() to learn
  # the parameters of W1, W2, b1, and b2 for this network and this data.
  def learn(self):
    def f(theta):
      return self.sparse_autoencoder(theta)
    theta = self.initialize_parameters()
    same_theta = theta.copy()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(f, theta,
                                           maxfun= self.options.max_iterations,
                                           iprint=1, m=20)
    W1, W2, b1, b2 = self.unflatten(x)
    utils.save_as_figure(W1.T, "%s/network.png" % self.options.output_dir)

    return SparseAutoEncoderSolution(W1, W2, b1, b2)
