import keras
from keras.applications.efficientnet_v2 import EfficientNetV2B1
import numpy as np
from keras import layers
import pandas as pd
import os
import tensorflow as tf
import glob
import pandas as pan
import warnings
warnings.filterwarnings('ignore')

def build_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    model = EfficientNetV2B1(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model

# Load image data
image_data = 'garbage_classification'
pd.DataFrame(os.listdir(image_data),columns=['Files_Name'])

files = [i for i in glob.glob(image_data + "//*//*")]
np.random.shuffle(files)
labels = [os.path.dirname(i).split("/")[-1] for i in files]
data = zip(files, labels)
dataframe = pan.DataFrame(data, columns = ["Image", "Label"])
dataframe

train_data_dir =image_data
batch_size = 128
target_size = (224,224)
validation_split = 0.2

train= tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=validation_split,
    subset="training",
    seed=50,
    image_size=target_size,
    batch_size=batch_size,
)
validation= tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=validation_split,
    subset="validation",
    seed=100,
    image_size=target_size,
    batch_size=batch_size,
)

class_names = train.class_names
class_names

model = build_model(12)
model.summary()

epochs = 1
hist = model.fit(train, epochs=epochs, validation_data=validation)

model.save('dimuthu.keras')