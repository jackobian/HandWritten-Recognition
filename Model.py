import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np

# ========================
# Part 1: Train & Save the Model
# ========================

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape and normalize data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

def create_improved_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create and compile the model
model = create_improved_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(train_images)

# Train the model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=40,
    validation_data=(test_images, test_labels)
)

# Save the model
model.save("improved_mnist_cnn.h5")
print("Model saved as improved_mnist_cnn.h5")

import matplotlib.pyplot as plt

# Визуализация процесса обучения
def plot_training_history(history):
    # График потерь (loss)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потери на обучающем наборе')
    plt.plot(history.history['val_loss'], label='Потери на тестовом наборе')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('График потерь')
    plt.legend()

    # График точности (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точность на обучающем наборе')
    plt.plot(history.history['val_accuracy'], label='Точность на тестовом наборе')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title('График точности')
    plt.legend()

    plt.show()

# Визуализируем историю обучения
plot_training_history(history)
