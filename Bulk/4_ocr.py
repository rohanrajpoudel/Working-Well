import cv2
from tensorflow.keras.models import load_model

# Load the trained OCR model
model = load_model("ocr_model.h5")

# Function to perform OCR on an image using the trained model
def perform_ocr(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to match the input size of the trained model
    image = cv2.resize(image, (32, 32))
    # Bilateral filtering
    image = cv2.bilateralFilter(image,5,30,45)
    # Otsu's thresholding
    _,image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thinning the image
    image = cv2.ximgproc.thinning(image)

    # Reshape the image to match the input shape expected by the model
    image = image.reshape(1, 32, 32, 1)

    # Use the model to predict the character
    prediction = model.predict(image)
    # Get the index of the predicted class
    predicted_class = prediction.argmax()

    # Convert the predicted class index to the corresponding character
    # This assumes that your classes are labeled with integer values corresponding to characters
    predicted_character = str(predicted_class)

    return predicted_character



# Example usage
image_path = "./ek.png"  # Update with the path to your image
predicted_character = perform_ocr(image_path)

print("Predicted Character using CNN OCR Model:", predicted_character)

