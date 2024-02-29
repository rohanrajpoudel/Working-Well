# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import os
import sys

def show_result(image_path,i,files_length,folder_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # filtered_image = cv2.GaussianBlur(image, (3, 3), 10)
    # Perform vertical edge detection
    sobel_kernel = np.array([[-3, 0, 3],
                             [-10, 0, 10],
                             [-3, 0, 3]])
    # Apply Sobel filter to compute horizontal gradient
    gradient_x = cv2.filter2D(image, -1, sobel_kernel)
    # Binarize the horizontal gradient using Otsu's method
    _, binary_vertical_map = cv2.threshold(np.abs(gradient_x).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Display the result using cv2.imshow
    cv2.imshow("Filtered Image", binary_vertical_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i+=1
    return i





folder_path = '../Our Dataset/Working Dataset/Num Plate/'
os.makedirs(folder_path, exist_ok=True)
all_files = os.listdir(folder_path)
jpg_files = [file for file in all_files if file.lower().endswith('.jpg')]
files_length=len(jpg_files)+1
i=1
for jpg_file in jpg_files:
    # Read an image for testing
    # image_path = dataset_path+str(i)+'.jpg'
    image_path = folder_path+jpg_file
    i= show_result(image_path,i,files_length,folder_path)
    if cv2.waitKey(33) == 27:
        sys.exit()