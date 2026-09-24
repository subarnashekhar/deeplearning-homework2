#!/usr/bin/env python3
"""Build and summarize a simple residual CNN for Question 5 Task 2."""

import tensorflow as tf


def residual_block(input_tensor, filters=64):
    """Apply two convolutions and add the original input through a skip path."""
    # The shortcut keeps the original information availble while the two
    # convolution layers learn a useful change, or residual, to add to it.
    shortcut = input_tensor
    x = tf.keras.layers.Conv2D(
        filters, kernel_size=(3, 3), padding="same", activation="relu"
    )(input_tensor)
    x = tf.keras.layers.Conv2D(
        filters, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)

    # Both tensors have the same shape because padding is "same", so they can
    # be added element by element before the final activation.
    x = tf.keras.layers.Add()([shortcut, x])
    return tf.keras.layers.Activation("relu")(x)


def build_resnet():
    """Create the ResNet-like model required for Question 5 Task 2."""
    # A smaller input keeps this architecure quick to summarize on a laptop.
    # The input layer defines the image shape that enters the network.
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    # The first convolution reduces spatial size while learning basic features.
    x = tf.keras.layers.Conv2D(
        64, kernel_size=(7, 7), strides=2, padding="same", activation="relu"
    )(inputs)
    # Repeated residual blocks let the network refine those learned feautres.
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    # Flatten converts the feature map into values for the dense classifier.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    # The final layer return probabilities for ten output classes.
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="simple_resnet")


def main():
    model = build_resnet()
    model.summary()


if __name__ == "__main__":
    main()
