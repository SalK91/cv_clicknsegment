import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os as os
import tkinter.messagebox

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

            # Initialize a flag to track whether an object is selected
            object_selected = False

            def mouse_callback(event, x, y, flags, param):
                nonlocal object_selected
                if event == cv.EVENT_LBUTTONDBLCLK:
                    for i in indexes.flatten():
                        (bx, by) = (boxes[i][0], boxes[i][1])
                        (bw, bh) = (boxes[i][2], boxes[i][3])
                        if bx < x < bx + bw and by < y < by + bh:
                            # Object selected, crop and display
                            cropped_object = clone[by:by + bh, bx:bx + bw]
                            cv.imshow("Cropped Object", cropped_object)
                            object_selected = True

            # Set up the mouse callback
            cv.namedWindow("YOLO Output")
            cv.setMouseCallback("YOLO Output", mouse_callback)

            while True:
                # Display the YOLO output
                cv.imshow("YOLO Output", image)

                key = cv.waitKey(1) & 0xFF

                # Exit the loop if 'q' is pressed or an object is selected
                if key == ord("q") or object_selected:
                    break

            cv.destroyAllWindows()
        else:
            tk.messagebox.showerror("Error", "Please select an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
