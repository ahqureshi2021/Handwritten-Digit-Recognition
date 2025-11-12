import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
                            layers.Dense(512, activation="relu"),
                            layers.Dense(10, activation="softmax")
                            ])

from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(784,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

train_images = train_images.reshape((60000, 784))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 784))
test_images = test_images.astype("float32") / 255

network.fit(train_images, train_labels, epochs=5, batch_size=128)

inp = test_images[1].reshape(1,784)

network.predict(inp).argmax()

test_digits = test_images[0:10]
predictions = network.predict(test_digits)


st.title("🖊️ Handwritten Digit Recognition (TensorFlow + Streamlit)")
st.write("Upload a **28x28 pixel** image of a handwritten number (0–9).")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L") 
    img = img.resize((28, 28)) 
    st.image(img, caption="Uploaded Image", width=150)

    img_array = np.array(img)/255
    img_array = img_array.reshape(1, 784)

    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    st.subheader(f"Predicted Digit: **{result}**")