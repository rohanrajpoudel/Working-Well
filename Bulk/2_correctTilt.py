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
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    try:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        # cv2.imshow("Outline", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
        # warped = (warped > T).astype("uint8") * 255
        # show the original and scanned images
        print("STEP 3: Apply perspective transform")
        # cv2.imshow("Original", imutils.resize(orig, height = 201))


        # cv2.imshow("Scanned", imutils.resize(warped, height = 201))
        # if cv2.waitKey(33) == 27:
        #     sys.exit()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(folder_path+'NP'+str(i)+'.jpg',orig)
        cv2.imwrite('../Our Dataset/Working Dataset/Corrected Tilt/TC_'+str(i)+'.jpg',warped)
    except:
        pass
        # cv2.imwrite(folder_path+'Dot/O'+str(i)+'.jpg',orig)
        # cv2.imshow("No counter found",imutils.resize(orig, height = 201))
        # if cv2.waitKey(33) == 27:
        #     sys.exit()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    i+=1
    if i==files_length:
        sys.exit()
    if cv2.waitKey(33) == 27:
        sys.exit()
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
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
# image = cv2.imread(args["image"])
