import cv2
import os
import sys

# Set the Dataset Path
dataset_path = "../Our Dataset/Working Dataset/Still Vehicle/"
#Check if the Directory exists
os.makedirs(dataset_path, exist_ok=True)

# Load all files in the folder
all_files = os.listdir(dataset_path)
# Filter out only the .jpg files
jpg_files = [file for file in all_files if file.lower().endswith('.jpg')]
# Get the length of the files
files_length = len(jpg_files)+1
i = 1

# Loop through all the .jpg files
for jpg_file in jpg_files:
    # Read an image for preProcess
    image_path = dataset_path+jpg_file
    # Read the image
    image = cv2.imread(image_path)
    # Rename file
    cv2.imwrite(dataset_path+f"SV_{i}.jpg", image)
    # remove the original file
    os.remove(image_path)
    i += 1
    if i==files_length:
        sys.exit()