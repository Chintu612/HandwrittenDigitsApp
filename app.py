import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


# Load the trained model
@st.cache_resource
def load_digit_model():
    return load_model('mnist_model.h5')


model = load_digit_model()


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to handle varying light conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the digit)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image to the bounding box
        digit = thresh[y:y + h, x:x + w]

        # Add padding
        padding = 20
        digit_padded = cv2.copyMakeBorder(digit, padding, padding, padding, padding,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize to 28x28
        digit_resized = cv2.resize(digit_padded, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize the image
        digit_normalized = digit_resized / 255.0

        # Reshape for model input
        digit_reshaped = digit_normalized.reshape(1, 28, 28, 1)

        return digit_reshaped
    else:
        st.error("No digit found in the image.")
        return None


st.title("Handwritten Digit Recognition")

# Create a radio button for selecting input method
input_method = st.radio("Select input method:", ("Upload Image", "Use Camera"))

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button('Preprocess and Predict'):
            preprocessed_image = preprocess_image(image, display=True)
            if preprocessed_image is not None:
                # Make prediction
                prediction = model.predict(preprocessed_image)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                st.success(f"Predicted Digit: {predicted_digit}")
                st.info(f"Confidence: {confidence:.2f}%")

elif input_method == "Use Camera":
    picture = st.camera_input("Take a picture")

    if picture:
        # Convert the file_bytes to opencv Image
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if st.button('Predict'):
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is not None:
                # Make prediction
                prediction = model.predict(preprocessed_image)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                st.success(f"Predicted Digit: {predicted_digit}")
                st.info(f"Confidence: {confidence:.2f}%")

st.markdown("---")
st.write("Note: This app works best with clear, centered images of single digits on a contrasting background.")