#!/usr/bin/env python3
"""Demonstrate 2-D convolution with different strides and padding modes."""

import numpy as np
import tensorflow as tf


# The input is the exact 5x5 matrix given in Question 3.
INPUT_MATRIX = np.array(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ],
    dtype=np.float32,
)

# This Laplacian-like kernel detects changes between a center value and its
# four neighbooring values. The center value is multiplied by -4.
KERNEL = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)


def apply_convolution(input_matrix, kernel, stride, padding):
    """Apply a 2-D TensorFlow convolution and return a plain NumPy matrix."""
    # TensorFlow expects image data in four dimmensions:
    # batch, height, width, and channels. This input has one image and one
    # channel, so we add those dimensions before calling the convolution.
    image = tf.convert_to_tensor(input_matrix[None, :, :, None], dtype=tf.float32)
    filter_tensor = tf.convert_to_tensor(kernel[:, :, None, None], dtype=tf.float32)

    # TensorFlow's convolution slides the kernel across the image. It usses
    # cross-correlation order, which is the standard behavior of Conv2D.
    output = tf.nn.conv2d(
        image,
        filters=filter_tensor,
        strides=[1, stride, stride, 1],
        padding=padding,
    )
    return output.numpy()[0, :, :, 0]


def print_feature_map(stride, padding, feature_map):
    """Print one result with a label that identifies its parameters."""
    print(f"Stride = {stride}, Padding = '{padding}'")
    print(feature_map)
    print()


def main():
    print("Input matrix:")
    print(INPUT_MATRIX)
    print("\nKernel:")
    print(KERNEL)
    print("\nOutput feature maps:\n")

    # VALID does not add zeros around the image, so the output is smaler.
    # SAME adds padding so the output size follows the stride setting.
    for stride, padding in ((1, "VALID"), (1, "SAME"), (2, "VALID"), (2, "SAME")):
        feature_map = apply_convolution(INPUT_MATRIX, KERNEL, stride, padding)
        print_feature_map(stride, padding, feature_map)


if __name__ == "__main__":
    main()
