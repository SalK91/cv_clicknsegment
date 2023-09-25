import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os as os
import tensorflow as tf

from math import hypot


## Functions
#CLICK CROP
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, image
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv.imshow("image", image)

def distance(p1,p2):
#"""Euclidean distance between two points."""
	xx1, yy1 = p1
	xx2, yy2 = p2
	return hypot(xx2 - xx1, yy2 - yy1)

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

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])

        if file_path:
            self.image_path = file_path
            image = Image.open(self.image_path)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

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
            image = cv.imread(self.image_path)
            clone = image.copy()
            (height, width) = image.shape[:2]

            # Determine YOLO output layer names
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # Create a blob from the input image and perform forward pass
            blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
                    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
                    cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv.namedWindow("image")
            cv.setMouseCallback("image", click_and_crop)

            # keep looping until double click
            while True:
                # display the image and wait for a keypress
                cv.imshow("image", image)
                key = cv.waitKey(1) & 0xFF
                #Double Click Break
                if(len(refPt)>=2):
                    break


            # if there are two reference points, then crop the region of interest from the image and display it
            print("Image Selected")


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
                cv.imshow("ROI", roi)
                cv.imwrite("CroppedImage.png", roi)

                cv.waitKey(0)


            # close all open windows
            cv.destroyAllWindows()
        else:
            tk.messagebox.showerror("Error", "Please select an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
