import tensorflow as tf
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('garbage_classification_model.h5')

# Load a test image
img_path = 'path_to_your_test_image.jpg'  # Update with the actual path to your test image
img = tf.keras.preprocessing.load_img(img_path, target_size=(224, 224))  # Resize to match model input size
img_array = tf.keras.preprocessing.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Preprocess the image (e.g., normalize pixel values)
img_array = img_array / 255.0  # Assuming normalization by dividing by 255 for RGB images

# Make predictions using the loaded model
predictions = loaded_model.predict(img_array)

# Assuming predictions is an array of probabilities for each class
print(predictions)
