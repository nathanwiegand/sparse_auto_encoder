#!/usr/bin/python2.4

import utils
import numpy as np
import scipy as sp
import scipy.io as sio
import sac

def normalize_data(data):
  data = data - np.mean(data)
  pstd = 3 * np.std(data)
  data = np.fmax(np.fmin(data, pstd), -pstd) / pstd
  data = (data + 1) * 0.4 + 0.1;
  return data

def sampleIMAGES(patchsize, num_patches):
  print "Called sampleIMAGES"
  IMAGES = sio.loadmat('data/IMAGES')['IMAGES']
  print "loaded images"
  patches = np.zeros([patchsize * patchsize, num_patches])
  print "zeroed patches"
  [ydim, xdim, num_images] = IMAGES.shape
  print "got shape", ydim, xdim, num_images

  for i in range(num_patches):
    img = np.random.randint(num_images)
    y_start = np.random.randint(ydim - patchsize + 1)
    x_start = np.random.randint(xdim - patchsize + 1)
    patch = IMAGES[y_start:y_start+patchsize, x_start:x_start+patchsize, img]
    patches[:,i] = patch.reshape([patchsize * patchsize])

  print "calling normalize"
  return normalize_data(patches)

def compute_numerical_gradient(fn, theta):
  epsilon = 1e-4
  numgrad = np.zeros(theta.shape)

  for i in range(theta.size):
    theta_minus = theta.copy()
    theta_plus = theta.copy()
    theta_minus[i] = theta_minus[i] - epsilon
    theta_plus[i] = theta_plus[i] + epsilon
    numgrad[i] = (fn(theta_plus) - fn(theta_minus)) / (2 * epsilon)

  return numgrad

patchsize = 8
num_patches = 10000
visible_size = patchsize * patchsize
hidden_size = 25
target_activation = 0.01
lamb = 0.0001
beta = 3

patches = sampleIMAGES(patchsize, num_patches)
options = sac.SparseAutoEncoderOptions(visible_size,
                                       hidden_size,
                                       learning_rate = lamb,
                                       beta = beta,
                                       sparsity = target_activation,
                                       output_dir = "ex4_output")
network = sac.SparseAutoEncoder(options, patches)


# def sal(theta):
#   return network.sparse_autoencoder(theta)
#
# theta = network.initialize_parameters()

#numgrad = compute_numerical_gradient(lambda x: sal(x)[0], theta)

# Eyeball the gradients
#print np.hstack([numgrad, grad])

#diff = linalg.norm(numgrad-grad) / linalg.norm(numgrad+grad)
#print "Normed difference: %f" % diff

network.learn()
