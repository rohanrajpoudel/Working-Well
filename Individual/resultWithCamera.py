import cv2
import torch
import os
import random
from tensorflow.keras.models import load_model
from picamera.array import PiRGBArray
from picamera import PiCamera

# Define the path to Faster RCNN Number Plate Detection Model
numPlateModelPath = '../Bulk/numberPlate-FasterRCNN.pth'
charSegModelPath = '../Bulk/characterSegment-FasterRCNN.pth'
ocrModelPath = '../Bulk/ocrModel.h5'
# Load the model
numPlateModel = torch.load(numPlateModelPath, map_location=torch.device('cpu'))
charSegModel = torch.load(charSegModelPath, map_location=torch.device('cpu'))
ocrModel = load_model(ocrModelPath)
# Set the model to evaluation mode
numPlateModel.eval()  
charSegModel.eval()  
# Define the class names
numPlateClasses = ["RedLP", "RedLP"]
charSegClasses = ["Char", "Char"]
camera = PiCamera()
rawCapture = PiRGBArray(camera)

# Function to perform OCR on an image using the trained model
def perform_ocr(image):
    try:
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Bilateral filtering
        image = cv2.bilateralFilter(image,5,30,45)
        # Otsu's thresholding
        _,image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thinning the image
        image = cv2.ximgproc.thinning(image)

        # Reshape the image to match the input shape expected by the model
        image = image.reshape(1, 32, 32, 1)

        # Use the model to predict the character
        prediction = ocrModel.predict(image)
        # Get the index of the predicted class
        predicted_class = prediction.argmax()

        # Convert the predicted class index to the corresponding character
        # This assumes that your classes are labeled with integer values corresponding to characters
        predicted_character = str(predicted_class)
        match predicted_character:
            case "10":
                predicted_character = "Baa"
            case "11":
                predicted_character = "Cha"
            case "12":
                predicted_character = "Pa"
            case "13":
                predicted_character = "Ga"
        return predicted_character
    except:
        pass

def charSegShow(image):
    image = cv2.resize(image,(400,200))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = charSegModel(image_tensor)

    # Extract relevant information from predictions
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    # Draw bounding boxes for "Char" class
    j = 0
    for box, score, label in zip(boxes, scores, labels):
        j += 1
        if charSegClasses[label] == "Char" and score > 0.8:  # Adjust the confidence threshold as needed
            box = [int(coord) for coord in box.tolist()]
            # image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            #to save the cropped image
            # print(box)
            cropped_image = image[(box[1]-4):(box[3]+4), (box[0]-2):(box[2]+2)]
            character = perform_ocr(cropped_image)
            # cv2.imwrite(f"./RedLP_{i}.jpg", cropped_image) 
            # image = cv2.putText(image, f"{score:.2f}", (box[0], box[1]),
            image = cv2.rectangle(image, (box[0], box[1]-4), (box[2], box[3]+4), (0, 255, 0), 2)
            image = cv2.putText(image, character, (box[0]+4, box[1]+13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def numPlateShow(image):
    # image = cv2.imread(image_path)
    h,w,_=image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = numPlateModel(image_tensor)

    # Extract relevant information from predictions
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    # Draw bounding boxes for "Red LP" class
    for box, score, label in zip(boxes, scores, labels):
        if numPlateClasses[label] == "RedLP" and score > 0.8:  # Adjust the confidence threshold as needed
            box = [int(coord) for coord in box.tolist()]
            # Corp the number plate
            cropped_image = image[(box[1]):(box[3]), (box[0]):(box[2])]
            # Draw rectangle
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            image = cv2.putText(image, f"LP: {score:.2f}", (box[0]-5, box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    image = cv2.resize(image,(int(w/1.5),int(h/1.5)))
    try:
        return cropped_image
    except:
        return image

def main():
    originalImage, finalResultImage = capture_image()
    cv2.imshow("Original Image", originalImage)
    cv2.imshow("Final Result", finalResultImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def capture_image():
    camera.capture(rawCapture, format="bgr")
    cap = rawCapture.array
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    try:
        numPlateImage = numPlateShow(cap)
        finalResultImage = charSegShow(numPlateImage)
        return cap, finalResultImage
    except:
        pass

# main()

