import cv2
import numpy as np
import os
import sys

def mulasaag(image_path,i):
    # Load the image
    src = cv2.imread(image_path)
    src = cv2.resize(src, (400, 200))
    # Convert to grayscale
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    
    # Applying thresholding technique 
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY) 
    
    # Using cv2.split() to split channels  
    # of coloured image 
    b, g, r = cv2.split(src) 
    
    # Making list of Red, Green, Blue 
    # Channels and alpha 
    rgba = [b, g, r, alpha] 
    
    # Using cv2.merge() to merge rgba 
    # into a coloured/multi-channeled image 
    dst = cv2.merge(rgba, 4) 

    # Display the extracted license plate
    cv2.imshow('License Plate', dst)
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
    i= mulasaag(image_path,i)
    if cv2.waitKey(33) == 27:
        sys.exit()