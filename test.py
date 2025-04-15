import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model_path = "model_weights.h5.keras"  # Adjust path if needed
model = load_model(model_path)

# Load and preprocess the image
image_path = "Sufyan Beg.jpg"  # Change this to your image path
img_size = (56, 56)  # Resize to match model input

img = load_img(image_path, target_size=img_size, color_mode="grayscale")
img_array = img_to_array(img)  # Convert to NumPy array
img_array = img_array / 255.0  # Normalize if required
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(img_array)

# Print raw output
print("Raw Model Output:", predictions)

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Get the predicted emotion
predicted_class = np.argmax(predictions)  # Get the index of the highest probability
predicted_emotion = emotion_labels[predicted_class]

print("Predicted Emotion:", predicted_emotion)
