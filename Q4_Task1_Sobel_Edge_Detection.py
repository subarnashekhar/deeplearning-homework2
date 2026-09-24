#!/usr/bin/env python3
"""Apply Sobel-X and Sobel-Y filters to a grayscale image."""

import argparse
import os

import cv2
import numpy as np


# These are the Sobel filters specified in Question 4 Task 1.
SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)
SOBEL_Y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float32,
)


def create_sample_image():
    """Create a small grayscale image when no external image is provided."""
    # A simple image with shapes makes the edges easy to see and keeps the
    # solution self-contained, so no extra image download is needed.
    image = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (135, 190), 180, thickness=-1)
    cv2.circle(image, (230, 105), 65, 255, thickness=-1)
    cv2.line(image, (25, 215), (290, 215), 120, thickness=8)
    return image


def load_image(image_path):
    """Load a grayscale image or create the included sample image."""
    if image_path:
        # The flag tells OpenCV to read one intensity value per pixel.
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image

    return create_sample_image()


def apply_sobel_filters(image):
    """Return the original image and normalized Sobel-X and Sobel-Y images."""
    # filter2D slides each filter over the image and calculats a weighted sum
    # for every pixel. The two filters highlight edges in different directions.
    sobel_x = cv2.filter2D(image, cv2.CV_32F, SOBEL_X)
    sobel_y = cv2.filter2D(image, cv2.CV_32F, SOBEL_Y)

    # Sobel results can contain negative values, so normalise them for display.
    sobel_x_display = cv2.convertScaleAbs(sobel_x)
    sobel_y_display = cv2.convertScaleAbs(sobel_y)
    return image, sobel_x_display, sobel_y_display


def save_results(images, output_dir):
    """Save the three images and return their file paths."""
    os.makedirs(output_dir, exist_ok=True)
    names = ("original_image.png", "sobel_x_edges.png", "sobel_y_edges.png")
    paths = []
    for name, image in zip(names, images):
        path = os.path.join(output_dir, name)
        if not cv2.imwrite(path, image):
            raise OSError(f"Could not save image: {path}")
        paths.append(path)
    return paths


def display_results(images):
    """Display the original image and both directional edge images."""
    # The three windows make it possible to compare the input and each edeg
    # direction side by side. Press any key in a window to close them.
    titles = ("Original Image", "Edge Detection using Sobel-X", "Edge Detection using Sobel-Y")
    for title, image in zip(titles, images):
        cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Sobel edge detection with OpenCV")
    parser.add_argument(
        "--image-path",
        default="",
        help="Optional path to a grayscale image; otherwise a sample is generated",
    )
    parser.add_argument(
        "--output-dir",
        default="q4_task1_outputs",
        help="Directory where the three result images are saved",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save images without opening display windows",
    )
    args = parser.parse_args()

    image = load_image(args.image_path)
    results = apply_sobel_filters(image)
    paths = save_results(results, args.output_dir)

    print(f"Loaded grayscale image with shape: {image.shape}")
    print("Saved images:")
    for path in paths:
        print(f"- {path}")

    if not args.no_display:
        display_results(results)


if __name__ == "__main__":
    main()
