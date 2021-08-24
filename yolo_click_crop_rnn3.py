# import the necessary packages
import numpy as np
import argparse
import time
import cv2
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

#import os
#os.system("python yourfile.py")


## FUNCTIONS

#CLICK LOCATION:
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

#CLICK COUNTER
count=0
def counter(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        count = count +1

#Click and Annotate
def marker(event, x, y, flags, param):
	global markers
	global stop

	if event == cv2.EVENT_LBUTTONDBLCLK:
		center = (x,y)
		markers.append([x, y])
		radius = 5
		cv2.circle(param, center, radius, (0,0,255), 2)
		isClosed = False
		thickness = 1
		color = (0, 255, 0)
		cv2.polylines(param,[np.array(markers)],isClosed, color,thickness)
		print(markers)

	if event == cv2.EVENT_RBUTTONDBLCLK:
		stop=1

def distance(p1,p2):
#"""Euclidean distance between two points."""
	xx1, yy1 = p1
	xx2, yy2 = p2
	return hypot(xx2 - xx1, yy2 - yy1)


#region Yolo Inference

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
clone = image.copy()

(H, W) = image.shape[:2]
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []
centers = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			centers.append((centerX, centerY))
			confidences.append(float(confidence))
			classIDs.append(classID)


# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])


# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#endregion
 


cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
    #Double Click Break

	if(len(refPt)>=2):
		cv2.waitKey(1)
		cv2.destroyWindow("image")
		break


#import time
#print("something")
#time.sleep(1)    # Pause 5.5 seconds
#print("something")

#region crop to nearest bounding box 
if len(refPt) >= 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	x = refPt[0][0]
	y = refPt[0][1]
	click_pos = (x, y)
	distance = [distance(click_pos, x) for x in centers]
	index = distance.index(min(distance))

	print(x, y)
	print(index)
	print(LABELS[classIDs[index]])
	print()

    #CROP
	x = boxes[index][0]
	y = boxes[index][1]
	w = boxes[index][2]
	h = boxes[index][3]
	hw = max(w, h)


	EnlargeFactor = 1.1
	x1 = round(max(x+w/2-EnlargeFactor*hw/2, 1))
	y1 = round(max(y+h/2-EnlargeFactor*hw/2, 1))
	x2 = round(x1 + EnlargeFactor*hw)
	y2 = round(y1 + EnlargeFactor*hw)

	ret = [(x1,y1)]
	ret.append((x2, y2))
	roi = clone[ret[0][1]:ret[1][1], ret[0][0]:ret[1][0]]
#endregion


markers = []
stop    = 0
cv2.namedWindow("dot")
cv2.setMouseCallback("dot", marker, param=roi)
while True:
	cv2.imshow("dot", roi)
	key = cv2.waitKey(1) & 0xFF
	if stop==1:
		cv2.waitKey(1)
		cv2.destroyWindow("dot")
		break

print(markers)
 

# close all open windows
# cv2.destroyAllWindows()


#region  Poly RNN Inference

from PIL import Image
im = Image.open('CroppedImage.png')
width, height = im.size
print(width, height)
newsize = (224, 224)
im1 = im.resize(newsize)
im1 = im1.save("image_resize.png")

##
#External PATHS
PolyRNN_metagraph='./polyrnn/models/poly/polygonplusplus.ckpt.meta'
PolyRNN_checkpoint='./polyrnn/models/poly/polygonplusplus.ckpt'
EvalNet_checkpoint='./polyrnn/models/evalnet/evalnet.ckpt'
GGNN_metagraph='./polyrnn/models/ggnn/ggnn.ckpt.meta'
GGNN_checkpoint='./polyrnn/models/ggnn/ggnn.ckpt'
#Const
_BATCH_SIZE=1
_FIRST_TOP_K = 6

# Creating the graphs
evalGraph = tf.Graph()
polyGraph = tf.Graph()
ggnnGraph = tf.Graph()
#Initializing and restoring the evaluator net.
with evalGraph.as_default():
    with tf.variable_scope("discriminator_network"):
        evaluator = EvalNet(_BATCH_SIZE)
        evaluator.build_graph()
    saver = tf.train.Saver()

    # Start session
    evalSess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
    ), graph=evalGraph)
    saver.restore(evalSess, EvalNet_checkpoint)

#Initializing and restoring PolyRNN++
model = PolygonModel(PolyRNN_metagraph, polyGraph)
model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))
polySess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True
), graph=polyGraph)
model.saver.restore(polySess, PolyRNN_checkpoint)


#Initializing and restoring GGNN
ggnnGraph = tf.Graph()
ggnnModel = GGNNPolygonModel(GGNN_metagraph, ggnnGraph)
ggnnSess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True
), graph=ggnnGraph)

ggnnModel.saver.restore(ggnnSess,GGNN_checkpoint)

#INPUT IMG CROP (224x224x3) -> object should be centered
crop_path='image_resize.png'

#Testing
image_np = io.imread(crop_path)
image_np = image_np[:,:,:3]
image_np = np.expand_dims(image_np, axis=0)
preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]

# sort predictions based on the eval score to pick the best.
preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)
#endregion

#Visualizing TOP_K and scores
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,3)
axes=np.array(axes).flatten()
[vis_polys(axes[i], image_np[0], np.array(pred['polys'][0]), title='score=%.2f' % pred['scores'][0]) for i,pred in enumerate(preds)]
plt.draw()
plt.pause(10) # pause how many seconds
plt.close()

#Let's run GGNN now on the bestPoly
bestPoly = preds[0]['polys'][0]
feature_indexs, poly, mask = utils.preprocess_ggnn_input(bestPoly)
preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
refinedPoly=preds_gnn['polys_ggnn']

#print(preds)
#Visualize the final prediction
#fig, ax = plt.subplots(1,1)
#vis_polys(ax,image_np[0],refinedPoly[0], title='PolygonRNN++')
#plt.show()

image =image_np[0]
rnn_markers =  refinedPoly[0]

h, w = image.shape[:2]
rnn_markers[:, 0] = np.round(rnn_markers[:, 0] * w,0)
rnn_markers[:, 1] = np.round(rnn_markers[:, 1] * h,0)
rnn_markers = rnn_markers.tolist()
rnn_markers = [[int(x) for x in sublist] for sublist in rnn_markers]
print(rnn_markers)
print(image)
#print(type(rnn_markers))
#print(rnn_markers.shape)

count=0
cv2.namedWindow("RNN")
cv2.setMouseCallback("RNN", counter)




while True:
	# display the image and wait for a keypress
	cv2.imshow("RNN", roi)
	isClosed = False
	thickness = 2
	color = (0, 255, 0)
	cv2.polylines(roi,[np.array(rnn_markers, dtype=np.int32)],isClosed, color,thickness)
	for point in rnn_markers:
		cv2.circle(roi,tuple(point),1,(0,0,255))

	key = cv2.waitKey(1) & 0xFF
    #Double Click Break

	if(count>=2):
		cv2.waitKey(1)
		cv2.destroyWindow("RNN")
		break
 

## C:/Users/salmansaeed.khan/.conda/envs/PolyRNN/python.exe d:/Personal-GIT/mlpractise/ComputerVision/Yolo/yolo_click_crop_rnn3.py --image image_0.jpg --yolo yolo-coco



