import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os as os
import tensorflow as tf
from math import hypot


from polyrnn.src.PolygonModel import PolygonModel
from polyrnn.src.EvalNet import EvalNet
from polyrnn.src.GGNNPolyModel import GGNNPolygonModel
from poly_utils import vis_polys
import skimage.io as io


## Functions
#CLICK CROP


class ImageUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Uploader")

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.open_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        self.yolo_button = tk.Button(root, text="Run YOLO", command=self.run_yolo)
        self.yolo_button.pack(pady=10)

        self.polyrnn_button = tk.Button(root, text="Run Polyrnn", command=self.run_polyrnn)
        self.polyrnn_button.pack(pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])

        if file_path:
            self.image_path = file_path
            image = Image.open(self.image_path)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo


    refPt = []
    cropping = False
    def click_and_crop(self,event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            cropping = True
        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            cropping = False
            # draw a rectangle around the region of interest
            cv.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv.imshow("image", self.image)

    def distance(self,xx1, yy1,xx2, yy2):
    #"""Euclidean distance between two points."""

        return hypot(xx2 - xx1, yy2 - yy1)


    def run_yolo(self):
        if hasattr(self, 'image_path'):

            # Input parameters
            args = {
                "yolo": 'yolo-coco',         # Directory containing YOLO model files
                "confidence": 0.5,           # Minimum confidence for object detection
                "threshold": 0.5             # Threshold for non-maximum suppression
            }

            # Load COCO class labels
            labels_path = os.path.sep.join([args["yolo"], "coco.names"])
            labels = open(labels_path).read().strip().split("\n")

            # Initialize colors for class labels
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

            # Paths to YOLO weights and model configuration
            weights_path = os.path.sep.join([args["yolo"], "yolov3.weights"])
            config_path = os.path.sep.join([args["yolo"], "yolov3.cfg"])

            # Load YOLO object detector
            net = cv.dnn.readNetFromDarknet(config_path, weights_path)

            # Load input image and get its dimensions
            self.image = cv.imread(self.image_path)
            clone = self.image.copy()
            (height, width) = self.image.shape[:2]

            # Determine YOLO output layer names
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # Create a blob from the input image and perform forward pass
            blob = cv.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layer_outputs = net.forward(output_layers)
 
            # Initialize lists for detected objects
            boxes = []
            confidences = []
            class_ids = []
            centers = []

            # Loop over each layer output
            for output in layer_outputs:
                # Loop over each detection
                for detection in output:
                    # Extract class ID and confidence
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Filter out weak predictions
                    if confidence > args["confidence"]:
                        # Scale bounding box coordinates
                        box = detection[0:4] * np.array([width, height, width, height])
                        (center_x, center_y, box_width, box_height) = box.astype("int")
                        x = int(center_x - (box_width / 2))
                        y = int(center_y - (box_height / 2))

                        # Update lists
                        boxes.append([x, y, int(box_width), int(box_height)])
                        centers.append((center_x, center_y))
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to suppress overlapping bounding boxes
            indexes = cv.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

            # Ensure at least one detection exists
            if len(indexes) > 0:
                for i in indexes.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in colors[class_ids[i]]]
                    cv.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
                    cv.putText(self.image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv.namedWindow("image")
            cv.setMouseCallback("image", self.click_and_crop)

            # keep looping until double click
            while True:
                # display the image and wait for a keypress
                cv.imshow("image", self.image)
                key = cv.waitKey(1) & 0xFF
                #Double Click Break
                if(len(self.refPt)>=2):
                    break

            

            # if there are two reference points, then crop the region of interest from the image and display it
            print("Image Selected")


            if len(self.refPt) >= 2:
                roi = clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
                x = self.refPt[0][0]
                y = self.refPt[0][1]
                click_pos = (x, y)
                distance = [self.distance(x,y,c_x,c_y) for (c_x,c_y) in centers]
                index = distance.index(min(distance))

                #print(x, y)
                #print(index)
                #print(LABELS[classIDs[index]])
                #print()

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
                cv.imwrite("CroppedImage.png", roi)
                cv.destroyWindow("image")
                #cv.imshow("ROI", roi)
                
                #cv.waitKey(0)

                file_path = "CroppedImage.png"

                if file_path:
                    self.croppedimage = file_path
                    image = Image.open(self.croppedimage)
                    self.photo = ImageTk.PhotoImage(image)
                    self.image_label.config(image=self.photo)
                    self.image_label.image = self.photo


            # close all open windows
            cv.destroyAllWindows()
        else:
            tk.messagebox.showerror("Error", "Please select an image first.")



    def run_polyrnn(self):
        if hasattr(self, 'croppedimage'):
            im = Image.open(self.croppedimage)
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
            print(preds)
            # draw polygon on the image
            # Draw the polygon on the image
            image = cv.imread(crop_path)

            bestPoly = preds[0]['polys'][0]
            bestPoly[:,0] = bestPoly[:,0] * image.shape[1]
            bestPoly[:,1] = bestPoly[:,1] * image.shape[0]
            bestPoly = [np.array(bestPoly, dtype=np.int32)]
            # Draw the polygon on the image
            cv.polylines(image, bestPoly, isClosed=True, color=(0, 0, 255), thickness=2)

            # write image to disk
            cv.imwrite("annoatedImage.png", image)

            #show image
            file_path = "annoatedImage.png"

            if file_path:
                self.croppedimage = file_path
                image = Image.open(self.croppedimage)
                self.photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo

            print(preds)
        else:
            tk.messagebox.showerror("Error", "Please select one object first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()


#polyrnn_button
#& C:/Users/salmank/anaconda3/envs/py36rnn/python.exe c:/Users/salmank/Documents/cv_clicknsegment/annotation5.py