import cv2
import torch
import sys



def calc_and_show_result(image_path, model, classes, i, files_length):
    image = cv2.imread(image_path)
    h,w,_=image.shape
    image = cv2.resize(image,(int(w/1.5),int(h/1.5)))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract relevant information from predictions
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    # Draw bounding boxes for "Red LP" class
    for box, score, label in zip(boxes, scores, labels):
        if classes[label] == "RedLP" and score > 0.8:  # Adjust the confidence threshold as needed
            box = [int(coord) for coord in box.tolist()]
            # image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            #to save the cropped image
            # print(box)
            cropped_image = image[(box[1]):(box[3]), (box[0]):(box[2])]
            cv2.imwrite(f"../Our Dataset/Working Dataset/Num Plate/NumPlt_{i}.jpg", cropped_image) 
            cv2.imwrite(f"../Our Dataset/Working Dataset/Still Vehicle/Still_{i}.jpg", image) 
            # image = cv2.putText(image, f"{classes[label]}: {score:.2f}", (box[0]-5, box[1] - 10),
            # image = cv2.putText(image, f"LP: {score:.2f}", (box[0]-5, box[1] - 10),
            # image = cv2.putText(image, f"Lic Plt:{box[0]-5} {box[1]-10} {score:.2f}", (box[0], box[1] - 5),
                                # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result using cv2.imshow
    # cv2.imshow(str(i), image)
    i+=1
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if cv2.waitKey(33) == 27 or i==files_length:
        sys.exit()
    return i

def main():
    # Define the path to Faster RCNN Number Plate Detection Model
    numPlateModelPath = './Bulk/numberPlate-FasterRCNN.pth'
    # Load the model
    numPlateModel = torch.load(numPlateModelPath, map_location=torch.device('cpu'))
    # Set the model to evaluation mode
    numPlateModel.eval()  
    # Define the class names
    numPlateclasses = ["RedLP", "RedLP"]  # Adjust the class names based on your dataset