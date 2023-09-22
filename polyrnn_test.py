import numpy as np
import argparse
import time
import cv2 as cv2
import os
from math import hypot
import tensorflow as tf
import numpy as np
from polyrnn.src.PolygonModel import PolygonModel
from polyrnn.src.EvalNet import EvalNet
from polyrnn.src.GGNNPolyModel import GGNNPolygonModel
import utils
from poly_utils import vis_polys
import skimage.io as io
from utils_app import *
from PIL import Image
import matplotlib.pyplot as plt


# Image To Segment
im = Image.open('CroppedImage.png')
width, height = im.size
print(width, height)
newsize = (224, 224)
im1 = im.resize(newsize)
im1 = im1.save("image_resize.png")

# External PATHS
PolyRNN_metagraph = './polyrnn/models/poly/polygonplusplus.ckpt.meta'
PolyRNN_checkpoint = './polyrnn/models/poly/polygonplusplus.ckpt'
EvalNet_checkpoint = './polyrnn/models/evalnet/evalnet.ckpt'
GGNN_metagraph = './polyrnn/models/ggnn/ggnn.ckpt.meta'
GGNN_checkpoint = './polyrnn/models/ggnn/ggnn.ckpt'

# Constants
_BATCH_SIZE = 1
_FIRST_TOP_K = 6

# Creating TensorFlow graphs
evalGraph = tf.Graph()
polyGraph = tf.Graph()
ggnnGraph = tf.Graph()

# Initializing and restoring the evaluator neural network
with evalGraph.as_default():
    with tf.variable_scope("discriminator_network"):
        evaluator = EvalNet(_BATCH_SIZE)
        evaluator.build_graph()
    saver = tf.train.Saver()

    # Start TensorFlow session
    evalSess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
    ), graph=evalGraph)
    saver.restore(evalSess, EvalNet_checkpoint)

# Initializing and restoring PolyRNN++
model = PolygonModel(PolyRNN_metagraph, polyGraph)
model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))
polySess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True
), graph=polyGraph)
model.saver.restore(polySess, PolyRNN_checkpoint)

# Initializing and restoring GGNN (Graph Neural Network)
ggnnGraph = tf.Graph()
ggnnModel = GGNNPolygonModel(GGNN_metagraph, ggnnGraph)
ggnnSess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True
), graph=ggnnGraph)
ggnnModel.saver.restore(ggnnSess, GGNN_checkpoint)

# Input image cropping (224x224x3) - object should be centered
crop_path = 'image_resize.png'

# Testing PolyRNN
image_np = io.imread(crop_path)
image_np = image_np[:, :, :3]
image_np = np.expand_dims(image_np, axis=0)
preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]

# Sort predictions based on the evaluation score to pick the best.
preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)

# Visualizing the top predictions and scores
fig, axes = plt.subplots(2, 3)
axes = np.array(axes).flatten()
[vis_polys(axes[i], image_np[0], np.array(pred['polys'][0]), title='score=%.2f' % pred['scores'][0]) for i, pred in enumerate(preds)]
plt.draw()
plt.pause(10)  # Pause for 10 seconds
plt.close()

# Running GGNN on the bestPoly from PolyRNN
bestPoly = preds[0]['polys'][0]
feature_indexs, poly, mask = utils.preprocess_ggnn_input(bestPoly)
preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
refinedPoly = preds_gnn['polys_ggnn']

# Visualizing the final prediction
fig, ax = plt.subplots(1, 1)
vis_polys(ax, image_np[0], refinedPoly[0], title='PolygonRNN++')
plt.show()

# & C:/Users/salmank/anaconda3/envs/py36rnn/python.exe c:/Users/salmank/Documents/cv_clicknsegment/polyrnn_test.py