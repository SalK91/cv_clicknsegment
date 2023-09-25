import cv2
import numpy as np

# Create an image with a white background
image = np.zeros((400, 400, 3), dtype=np.uint8) + 255

# Define the array of polygon points (x, y)
polygon_points = np.array([[100, 100], [300, 100], [200, 300]], np.int32)

# Reshape the array into a format expected by OpenCV (1, N, 2)
polygon_points = polygon_points.reshape((-1, 1, 2))

# Convert the polygon_points array to a list of tuples
polygon_points_list = polygon_points.tolist()

print(polygon_points_list)
# Draw the polygon on the image
#cv2.polylines(image, [np.array(polygon_points_list, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

# Display the image with the polygon
cv2.imshow("Image with Polygon", image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()