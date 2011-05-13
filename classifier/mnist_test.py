import pickle
import sys
import numpy as np
import scipy.optimize
sys.path.append("..")
import utils
import sac
from utils import ASSERT_SIZE, ASSERT_NO_NAN

MAX_PATCHES = 60000
images = utils.load_images("../data/train-images-idx3-ubyte")
labels_ = utils.load_labels("../data/train-labels-idx1-ubyte")
patches = images[:, 0:MAX_PATCHES]
labels = labels_[0:MAX_PATCHES]

# Note, this is the output from running mnist_train.py in the top level.
print "Reading edge detector."
fname = "data/numeral_sac.pickle"
f = open(fname, "r")
edge_detector_solution = pickle.load(f)

options = sac.SparseAutoEncoderOptions(28 * 28,
                                       196,
                                       output_dir = "output")

edge_detector = sac.SparseAutoEncoder(options, patches)
print "Computing edges."
edges, identity = edge_detector.feed_forward(images[:, 0:MAX_PATCHES],
                                             edge_detector_solution.W1,
                                             edge_detector_solution.W2,
                                             edge_detector_solution.b1,
                                             edge_detector_solution.b2)

# Note, this is the output of running digit_classifier.py in this directory.
fname = "recognizer_network.pickle"
f = open(fname, "r")
digit_classifier_solution = pickle.load(f)

print digit_classifier_solution.W1.shape
print edges.shape
print digit_classifier_solution.b1.shape

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

z2 = np.dot(digit_classifier_solution.W1, edges) + digit_classifier_solution.b1
a2 = sigmoid(z2)
print a2

results = {}
results[True] = 0
results[False] = 0
for i in range(a2.shape[1]):
  results[labels_[i] == np.argmax(a2[:, i])] += 1

total = results[True] + results[False]
print "Total: ", total
print "Correct: %d (%f)" % (results[True], 1.0*results[True]/total)
print "Wrong: %d (%f)" % (results[False], 1.0*results[False]/total)
