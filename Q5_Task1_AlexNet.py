#!/usr/bin/env python3
"""Build and summarize a simplified AlexNet architecture."""

import tensorflow as tf


def build_alexnet():
    """Create the simplified AlexNet model required for Question 5 Task 1."""
    # AlexNet expects a color image with height, width, and three channnels.
    # The batch dimension is added automatically when the model is used.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(227, 227, 3)),
            # The first convolution learns basic patterns such as edges and
            # uses a large stride to reduce the image size quickly.
            tf.keras.layers.Conv2D(
                96, kernel_size=(11, 11), strides=4, activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            tf.keras.layers.Conv2D(
                256, kernel_size=(5, 5), activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            # Deeper convolution layers can learn more detailed visual featuers.
            tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            # Flatten changes the final feature maps into one long vector.
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation="relu"),
            # Dropout randomly disables some connections during training to
            # help prevent the model from memorizing the training images.
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            # Softmax produces probabilities for ten possible classes.
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="simplified_alexnet",
    )
    return model


def main():
    model = build_alexnet()
    model.summary()


if __name__ == "__main__":
    main()
