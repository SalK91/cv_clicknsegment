import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os as os
import tensorflow as tf

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

            cv.imshow("YOLO Output", image)
            cv.waitKey(0)
        else:
            tk.messagebox.showerror("Error", "Please select an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
