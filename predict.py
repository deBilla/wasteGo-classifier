import tensorflow as tf
import numpy as np

# Load the saved model

# loaded_model = tf.keras.Model("garbage_classification-final.keras")
loaded_model = tf.keras.models.load_model('dimuthu.keras')

image = tf.keras.utils.load_img('test.jpg', target_size=(224, 224))
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr
predictions = loaded_model.predict(input_arr)
print (predictions)

predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_class)