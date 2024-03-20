import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import glob
import warnings

warnings.filterwarnings('ignore')

# Load image data
image_data = 'garbage_classification'
files = [i for i in glob.glob(image_data + "//*//*")]
np.random.shuffle(files)
labels = [os.path.dirname(i).split("/")[-1] for i in files]
data = zip(files, labels)
dataframe = pd.DataFrame(data, columns=["Image", "Label"])

# Split data into train and validation sets
train_data_dir = image_data
batch_size = 128
target_size = (224, 224)
validation_split = 0.2

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=validation_split,
    subset="training",
    seed=50,
    image_size=target_size,
    batch_size=batch_size,
)
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=validation_split,
    subset="validation",
    seed=100,
    image_size=target_size,
    batch_size=batch_size,
)

# Load the downloaded EfficientNetV2 model weights
base_model = tf.keras.applications.EfficientNetV2B1(input_shape=(224, 224, 3), include_top=False, weights=None)
weights_path = 'efficientnetv2-b1_notop.h5'  # Update with the correct path
base_model.load_weights(weights_path)

keras_model = keras.models.Sequential([
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(12, activation=tf.nn.softmax)  # 12 classes
])
keras_model.summary()

# Compile the model
keras_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("my_keras_model.keras", save_best_only=True)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# Train the model
hist = keras_model.fit(train_ds, epochs=10, validation_data=validation_ds, callbacks=[checkpoint, early_stopping])

# Evaluate the model
val_loss, val_acc = keras_model.evaluate(validation_ds)
print('Validation Loss:', val_loss)
print('Validation Accuracy:', val_acc)

# Save the trained model
keras_model.save('garbage_classification_model.h5')

# Load the saved model for later use
loaded_model = keras.models.load_model('garbage_classification_model.h5')

# Example usage of the loaded model
# predictions = loaded_model.predict(test_data)