import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import utils

class SparseAutoEncoderOptions:
  ##############################################################################
  #  These network parameters are specified by by Andrew Ng specifically for the
  #  MNIST data set here:
  #     http://ufldl.stanford.edu/wiki/index.php/Exercise:Vectorization
  ##############################################################################
  def __init__(self,
               visible_size,
               hidden_size,
               sparsity = 0.1,
               learning_rate = 3e-3,
               beta = 3,
               output_dir = "output"):
    self.visible_size = visible_size
    self.hidden_size = hidden_size
    self.sparsity_param = sparsity
    self.learning_rate = learning_rate
    self.beta = beta
    self.output_dir = output_dir

class SparseAutoEncoder:
  def __init__(self, options, data):
    self.options = options
    self.data = data

    self.frame_number = 0

  def flatten(self, W1, W2, b1, b2):
    return np.array(np.hstack([W1.ravel('F'), W2.ravel('F'),
                               b1.ravel('F'), b2.ravel('F')]), order='F')

  def unflatten(self, theta):
    hidden_size = self.options.hidden_size
    visible_size = self.options.visible_size
    hv = hidden_size * visible_size
    W1 = theta[0:hv].reshape([hidden_size, visible_size], order='F')
    W2 = theta[hv:2*hv].reshape([visible_size, hidden_size], order='F')
    b1 = theta[2*hv:2*hv+hidden_size].reshape([hidden_size, 1], order='F')
    b2 = theta[2*hv+hidden_size:].reshape([visible_size, 1], order='F')
    return (W1, W2, b1, b2)

  def initialize_parameters(self):
    hidden_size = self.options.hidden_size
    visible_size = self.options.visible_size
    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random([hidden_size, visible_size]) * 2 * r - r;
    W2 = np.random.random([visible_size, hidden_size]) * 2 * r - r;
    b1 = np.zeros([hidden_size, 1])
    b2 = np.zeros([visible_size, 1])

    return self.flatten(W1, W2, b1, b2)

  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def sparse_autoencoder(self, theta):
    visible_size = self.options.visible_size
    hidden_size = self.options.hidden_size
    lamb = self.options.learning_rate
    rho = self.options.sparsity_param
    beta = self.options.beta

    x = self.data

    m = x.shape[1]

    W1, W2, b1, b2 = self.unflatten(theta)
    utils.save_as_figure(W1.T, "%s/w1frame%03d.png" % (self.options.output_dir,
                                                       self.frame_number))
    utils.save_as_figure(W2.T, "%s/w2frame%03d.png" % (self.options.output_dir,
                                                       self.frame_number))
    self.frame_number += 1

    m_ones = np.ones((1, m))
    B1 = np.dot(b1, m_ones)
    B2 = np.dot(b2, m_ones)

    # Forward pass
    z2 = np.dot(W1, x) + B1 # W1 * x + b1
    a2 = self.sigmoid(z2)
    z3 = np.dot(W2, a2) + B2 # W2 * a2 + b2
    a3 = self.sigmoid(z3)

    # Compute average activation
    rho_hat = np.mean(a2,1)[:, np.newaxis]
    kl = rho*np.log(rho/rho_hat) + (1-rho)*np.log((1-rho)/(1-rho_hat))

    cost = 1./(2.* m)*np.sum(np.sum((a3 - x)**2)) + \
           (lamb/2.)*(np.sum(np.sum(W1**2)) + np.sum(np.sum(W2**2))) + \
           beta*np.sum(kl)
    y = x # we're learning an identity function
    delta3 = -(y - a3) * a3*(1-a3)
    sparsity = np.dot((-rho/rho_hat + (1-rho)/(1-rho_hat)), m_ones)
    delta2 = (np.dot(W2.T, delta3) + beta * sparsity) * a2 * (1-a2)
    W2_grad = 1./m * np.dot(delta3, a2.T) + lamb * W2
    b2_grad = 1./m * np.sum(delta3, 1)[:, np.newaxis]
      # sum the rows of delta3 and then mult by  1/m
    W1_grad = 1./m * np.dot(delta2, x.T) + lamb * W1
    b1_grad = 1./m * np.sum(delta2, 1)[:, np.newaxis]

    grad = self.flatten(W1_grad, W2_grad, b1_grad, b2_grad)

    return (cost, grad)

  def learn(self):
    def s(theta):
      return self.sparse_autoencoder(theta)
    theta = self.initialize_parameters()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(s, theta, maxfun=100, iprint=1, m=20)
    W1, W2, b1, b2 = self.unflatten(x)
    utils.save_as_figure(W1.T, "%s/network.png" % self.options.output_dir)
