import pickle
import sys
import numpy as np
import scipy.optimize
sys.path.append("..")
import utils
import sac
from utils import ASSERT_SIZE, ASSERT_NO_NAN

MAX_PATCHES = 2000
images = utils.load_images("../data/train-images-idx3-ubyte")
# Label entries correspond 1-to-1 with the image file.
labels_ = utils.load_labels("../data/train-labels-idx1-ubyte")

patches = images[:, 0:MAX_PATCHES]
labels = labels_[0:MAX_PATCHES]
visible_size = 28*28
hidden_size = 196

options = sac.SparseAutoEncoderOptions(visible_size,
                                       hidden_size,
                                       output_dir="output",
                                       max_iterations = 400)

fname = "data/numeral_sac.pickle"
f = open(fname, "r")
solution = pickle.load(f)

network = sac.SparseAutoEncoder(options, patches)
theta = network.flatten(solution.W1, solution.W2, solution.b1, solution.b2)

i = 0

input_size = hidden_size # same as the hidden size of the SAC
output_size = 10 # one per digit

digit_options = sac.SparseAutoEncoderOptions(input_size,
                                             output_size,
                                             output_dir="output",
                                             max_iterations = 400)

digit_recognizer = sac.SparseAutoEncoder(digit_options, patches)

print patches.shape
recognized_patches = np.zeros((input_size, 0))
print "RECOGNIZED_PATCHES.shape", recognized_patches.shape

recognized_patches, identity = network.feed_forward(patches,
    solution.W1, solution.W2, solution.b1, solution.b2)

i = 0
label_matrix = np.zeros([output_size, MAX_PATCHES])
for label in labels:
  label_matrix[label, i] = 1
  i += 1

print "Done"
print "RECOGNIZED_PATCHES",recognized_patches
print "RECOGNIZED_PATCHES_MAX", np.max(recognized_patches)
print "RECOGNIZED_PATCHES.shape", recognized_patches.shape
print label_matrix

ASSERT_NO_NAN(label_matrix)
ASSERT_NO_NAN(recognized_patches)

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def classifier(data, labels, theta):
  """ This is just a simple neural network with no hidden layer.  The output is
  a 10 x 1 matrix of real numbers. Take the argmax of this to find the most
  likely digit.
  """
  lamb = 0.001

  x = data
  m = x.shape[1]

  ios = input_size * output_size
  W1 = theta[0:ios].reshape([output_size, input_size], order='F')
  b1 = theta[ios:].reshape([output_size, 1], order='F')
  ASSERT_SIZE(W1, (output_size, input_size))
  ASSERT_SIZE(b1, (output_size, 1))

  z2 = np.dot(W1, x) + b1
  a2 = sigmoid(z2)
  ASSERT_SIZE(a2, (output_size, m))
  y = labels

  cost = 0.5/m * np.sum((y-a2)**2) + (lamb/2.) * (np.sum(W1**2))

  delta2 = -(y - a2) * a2*(1-a2)
  ASSERT_SIZE(delta2, (output_size, m))

  # sum the rows of delta3 and then mult by  1/m
  W1_grad = 1./m * np.dot(delta2, x.T) + lamb * W1
  ASSERT_SIZE(W1_grad, (output_size, input_size))

  b1_grad = 1./m * np.sum(delta2, 1)[:, np.newaxis]
  ASSERT_SIZE(b1_grad, (output_size, 1))

  grad = np.array(np.hstack([W1_grad.ravel('F'), b1_grad.ravel('F'),]),
                  order='F')
  return (cost, grad)


def train(theta):
  return classifier(recognized_patches, label_matrix, theta)

r = np.sqrt(10) / np.sqrt(input_size + output_size + 1)
W1 = np.random.random([output_size, input_size]) * 2 * r - r;
b1 = np.zeros([output_size, 1])
theta = np.array(np.hstack([W1.ravel('F'), b1.ravel('F')]), order='F')
ASSERT_NO_NAN(theta)
print theta
x, f, d = scipy.optimize.fmin_l_bfgs_b(train, theta,
                                       maxfun=500,
                                       iprint=1, m=20)
print d
print "x=",x
ASSERT_NO_NAN(x)
ios = input_size * output_size
W1 = x[0:ios].reshape([output_size, input_size], order='F')
b1 = x[ios:].reshape([output_size, 1], order='F')
ASSERT_NO_NAN(W1)
ASSERT_NO_NAN(b1)
print "W1", W1
print "b1", b1

test_number = 0
test_patch = images[:, test_number]

features, identity = network.feed_forward(test_patch[:,np.newaxis],
                                          solution.W1, solution.W2,
                                          solution.b1, solution.b2)

z2 = np.dot(W1, features) + b1
a2 = sigmoid(z2)
ASSERT_SIZE(a2, (output_size, 1))

utils.save_as_figure(test_patch.T[:,np.newaxis], "output/test_patch.png")
utils.save_as_figure(features.T, "output/features.png")
print np.max(W1)
print "a2", a2


print "theta=", theta
print labels_[test_number]
print np.argmax(a2)

answer = sac.SparseAutoEncoderSolution(W1,None,b1,None)
output = open("recognizer_network.pickle", "w")
output.write(pickle.dumps(answer))
