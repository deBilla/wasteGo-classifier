import tensorflow as tf
import numpy as np
import requests
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

# Load the saved model
loaded_model = tf.keras.models.load_model('dimuthu.keras')

def preprocess_image(url):
    # Load image from URL and preprocess it
    response = requests.get(url)
    image = tf.keras.utils.load_img(BytesIO(response.content), target_size=(224, 224))
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch.
    return input_arr

@app.route('/predict', methods=['GET'])
def predict():
    # Get the image URL from the request
    image_url = request.args.get('image_url')

    if not image_url:
        return jsonify({'error': 'Image URL not provided'}), 400

    try:
        # Preprocess the image
        input_arr = preprocess_image(image_url)

        # Make predictions
        predictions = loaded_model.predict(input_arr)

        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1)[0]

        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
