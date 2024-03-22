import tensorflow as tf
import keras
from keras.applications.xception import Xception
from keras import layers
import numpy as np
import pandas as pd
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def build_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    base_model = Xception(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    base_model.trainable = False

    data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.2),
    ]
    )
    x = data_augmentation(base_model.output)
    avg = keras.layers.GlobalAveragePooling2D()(x)
    output = keras.layers.Dense(num_classes, activation = "softmax")(avg)
    model = keras.Model(inputs = base_model.input, outputs = output)
    optimizer = keras.optimizers.SGD(learning_rate = 0.2, momentum = 0.9)
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer,
              metrics = ["accuracy"])

    return model

# Load image data
image_data = 'garbage_classification'
pd.DataFrame(os.listdir(image_data),columns=['Files_Name'])

files = [i for i in glob.glob(image_data + "//*//*")]
np.random.shuffle(files)
labels = [os.path.dirname(i).split("/")[-1] for i in files]
data = zip(files, labels)
dataframe = pd.DataFrame(data, columns = ["Image", "Label"])
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

callbacks = [
    keras.callbacks.ModelCheckpoint(
    filepath="model-xception-best.keras",
    save_best_only=True,
    monitor="val_loss"),
    
    keras.callbacks.EarlyStopping(patience = 10,
                                  restore_best_weights = True)
]

epochs = 1
hist = model.fit(train, epochs=epochs, validation_data=validation)

model.save('model-xception.keras')