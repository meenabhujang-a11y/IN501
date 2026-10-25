import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def main():

    # --------------------------------------------------
    # Load MNIST dataset
    # --------------------------------------------------

    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Dataset loaded successfully.")

    print()
    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Test images:", x_test.shape)
    print("Test labels:", y_test.shape)

    # --------------------------------------------------
    # Display first 10 images before preprocessing
    # --------------------------------------------------

    plt.figure(figsize=(10, 4))

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        plt.imshow(
            x_train[i],
            cmap="gray"
        )

        plt.title(
            f"Digit: {y_train[i]}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # # --------------------------------------------------
    # # Normalize pixel values from 0-255 to 0-1
    # # --------------------------------------------------

    # x_train = x_train.astype("float32") / 255.0
    # x_test = x_test.astype("float32") / 255.0

    # # --------------------------------------------------
    # # Reshape images for Conv2D
    # # --------------------------------------------------

    # x_train = x_train.reshape(
    #     x_train.shape[0],
    #     28,
    #     28,
    #     1
    # )

    # x_test = x_test.reshape(
    #     x_test.shape[0],
    #     28,
    #     28,
    #     1
    # )

    # print()
    # print("After preprocessing:")

    # print(
    #     "Training images:",
    #     x_train.shape
    # )

    # print(
    #     "Test images:",
    #     x_test.shape
    # )


main()
