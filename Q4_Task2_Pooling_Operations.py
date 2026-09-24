#!/usr/bin/env python3
"""Demonstrate 2x2 max pooling and average pooling."""

import numpy as np
import tensorflow as tf


def main():
    # A fixed seed makes the random 4x4 matrix reproducible for each run.
    tf.random.set_seed(7)
    input_matrix = tf.random.uniform(
        shape=(1, 4, 4, 1), minval=0, maxval=10, dtype=tf.float32
    )

    # Pooling looks at small windows and reduses the amount of spatial data.
    # Max pooling keeps the largest value in each window, while average pooling
    # calculates the mean value. Both use a 2x2 window moving two steps.
    max_pool = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )
    average_pool = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )

    max_pooled_matrix = max_pool(input_matrix)
    average_pooled_matrix = average_pool(input_matrix)

    # Remove the batch and chanel dimensions so the printed output is easier
    # for a begginer to compare with the original 4x4 matrix.
    print("Original 4x4 matrix:")
    print(input_matrix.numpy()[0, :, :, 0])
    print("\nMax-pooled matrix:")
    print(max_pooled_matrix.numpy()[0, :, :, 0])
    print("\nAverage-pooled matrix:")
    print(average_pooled_matrix.numpy()[0, :, :, 0])


if __name__ == "__main__":
    main()
