#!/usr/bin/python

import pickle
import utils
import sac
import sys

if len(sys.argv) != 2:
  print "Usage: ./display_saved_network.py somefile.pickle"
  sys.exit(1)

fname = sys.argv[1]
f = open(fname, "r")
solution = pickle.load(f)

utils.save_as_figure((solution.W1 + solution.b1).T, "loadedW1.png")
utils.save_as_figure(solution.W2, "loadedW2.png")


images = utils.load_images("data/train-images-idx3-ubyte")
labels = utils.load_labels("data/train-labels-idx1-ubyte")
utils.save_as_figure(images[:, 0:100], "output/input.png")

patches = images[:, 0:10000]
visible_size = 28*28
hidden_size = 196


options = sac.SparseAutoEncoderOptions(visible_size,
                                       hidden_size,
                                       output_dir="output",
                                       max_iterations = 400)

network = sac.SparseAutoEncoder(options, patches)

theta = network.flatten(solution.W1, solution.W2, solution.b1, solution.b2)

#print network.sparse_autoencoder(theta)
