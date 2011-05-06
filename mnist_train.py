#!/usr/bin/python

import utils
import numpy as np
import scipy.optimize
import struct
import sac

images = utils.load_images("data/train-images-idx3-ubyte")
labels = utils.load_labels("data/train-labels-idx1-ubyte")
utils.save_as_figure(images[:, 0:100], "output/input.png")

patches = images[:, 0:10000]
visible_size = 28*28
hidden_size = 196

options = sac.SparseAutoEncoderOptions(visible_size,
                                       hidden_size,
                                       output_dir="output")
network = sac.SparseAutoEncoder(options, patches)
answer = network.learn()
