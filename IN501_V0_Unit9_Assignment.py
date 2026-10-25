import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def main():

    # --------------------------------------------------
    # 1. Load the MNIST dataset
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
    # 2. Display the first 10 training images
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

    # --------------------------------------------------
    # 3. Preprocess the dataset
    # --------------------------------------------------

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape images to include one grayscale channel
    x_train = x_train.reshape(
        -1,
        28,
        28,
        1
    )

    x_test = x_test.reshape(
        -1,
        28,
        28,
        1
    )

    print()
    print("Data preprocessing completed.")
    print("Training data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)

    # --------------------------------------------------
    # 4. Create the Sequential CNN model
    # --------------------------------------------------

    model = Sequential()

    # First convolution layer
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        )
    )

    # First pooling layer
    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # Second convolution layer
    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        )
    )

    # Second pooling layer
    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # Convert feature maps into a one-dimensional vector
    model.add(
        Flatten()
    )

    # Fully connected hidden layer
    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    # Output layer for digits 0 through 9
    model.add(
        Dense(
            10,
            activation="softmax"
        )
    )

    # --------------------------------------------------
    # 5. Compile the model
    # --------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print()
    print("CNN Model Summary:")
    model.summary()

    # --------------------------------------------------
    # 6. Train the model
    # --------------------------------------------------

    print()
    print("Training the CNN model...")

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1
    )

    # --------------------------------------------------
    # 7. Evaluate the model on test data
    # --------------------------------------------------

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print()
    print("====================================")
    print("          TEST RESULTS")
    print("====================================")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Accuracy %:    {test_accuracy * 100:.2f}%")

    # --------------------------------------------------
    # 8. Generate predictions from test data
    # --------------------------------------------------

    predictions = model.predict(
        x_test,
        verbose=0
    )

    # Get the digit with the highest probability
    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    # --------------------------------------------------
    # 9. Print predictions for first 10 test images
    # --------------------------------------------------

    print()
    print("====================================")
    print("      FIRST 10 PREDICTIONS")
    print("====================================")

    for i in range(10):

        print(
            f"Image {i + 1}: "
            f"Actual = {y_test[i]}, "
            f"Predicted = {predicted_labels[i]}"
        )

    # --------------------------------------------------
    # 10. Display first 10 test images with predictions
    # --------------------------------------------------

    plt.figure(figsize=(12, 5))

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        plt.imshow(
            x_test[i].reshape(28, 28),
            cmap="gray"
        )

        plt.title(
            f"Actual: {y_test[i]}\n"
            f"Predicted: {predicted_labels[i]}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

    print()
    print("MNIST CNN analysis completed.")


main()
