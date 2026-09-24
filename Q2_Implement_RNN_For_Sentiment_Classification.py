#!/usr/bin/env python3
"""Train an LSTM sentiment classifier on the Keras IMDB dataset."""

import argparse
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def load_and_prepare_data(
    vocabulary_size, sequence_length, max_train_samples, max_test_samples
):
    """Load IMDB reviews, limit the data, and pad every review to one length."""
    # IMDB reviews are already represented as lists of word IDs. The lable is
    # 0 for negative and 1 for positive, so this is a binary classification task.
    (train_reviews, train_labels), (test_reviews, test_labels) = (
        tf.keras.datasets.imdb.load_data(num_words=vocabulary_size)
    )

    # Fewer reviews make the experiment quicker on a laptop. Using None keeps
    # the complete original split when the user wants a larger experiment.
    if max_train_samples is not None:
        train_reviews = train_reviews[:max_train_samples]
        train_labels = train_labels[:max_train_samples]
    if max_test_samples is not None:
        test_reviews = test_reviews[:max_test_samples]
        test_labels = test_labels[:max_test_samples]

    # Reviews have different lengths, but a neural network batch needs one
    # fixed shape. Padding adds zeros to short reviews and truncates long ones.
    train_reviews = tf.keras.utils.pad_sequences(
        train_reviews, maxlen=sequence_length, padding="post", truncating="post"
    )
    test_reviews = tf.keras.utils.pad_sequences(
        test_reviews, maxlen=sequence_length, padding="post", truncating="post"
    )

    return (
        train_reviews,
        np.asarray(train_labels),
        test_reviews,
        np.asarray(test_labels),
    )


def build_model(vocabulary_size, embedding_size, lstm_units):
    """Build an embedding plus LSTM model for positive/negative prediction."""
    # The Embedding layer turns each word ID into a learned vector of numbers.
    # The LSTM reads those vectors in order and keeps usefull review context.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(None,)),
            tf.keras.layers.Embedding(vocabulary_size, embedding_size, mask_zero=True),
            tf.keras.layers.LSTM(lstm_units),
            # One sigmoid output is a probability that the review is positive.
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # Command-line options let the experimant use smaller data on a laptop.
    parser = argparse.ArgumentParser(
        description="Classify IMDB reviews with an LSTM RNN"
    )
    parser.add_argument("--vocabulary-size", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Number of training reviews; use 0 for the complete training split",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=5000,
        help="Number of test reviews; use 0 for the complete test split",
    )
    args = parser.parse_args()

    # Fixed seeds makes repeated runs easier to compare across experiments.
    # Fixed seeds make repeated runs more comparable, although some hardware
    # and TensorFlow operations can still introduce small differences.
    random.seed(7)
    np.random.seed(7)
    tf.random.set_seed(7)

    max_train_samples = args.max_train_samples or None
    max_test_samples = args.max_test_samples or None
    # The data loader prepares both splits with the same sequence length.
    train_reviews, train_labels, test_reviews, test_labels = load_and_prepare_data(
        args.vocabulary_size,
        args.sequence_length,
        max_train_samples,
        max_test_samples,
    )
    print(
        f"Training reviews: {len(train_reviews):,}; "
        f"test reviews: {len(test_reviews):,}; "
        f"sequence length: {args.sequence_length}"
    )

    model = build_model(
        args.vocabulary_size, args.embedding_size, args.lstm_units
    )
    # The model summary shows the layers and parameter count before training.
    model.summary()

    # During training, the model compares its probablity with the known label
    # and updates its weights to make future predictions more accurate.
    model.fit(
        train_reviews,
        train_labels,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    # A probability of at least 0.5 is classified as positive; otherwise it is
    # classified as negative. Keeping probabilities first makes this threshold
    # easy to change for a different precision-recall tradeoff.
    predicted_probabilities = model.predict(
        test_reviews, batch_size=args.batch_size, verbose=0
    ).ravel()
    predicted_labels = (predicted_probabilities >= 0.5).astype(np.int32)

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(test_labels, predicted_labels, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["negative", "positive"],
            digits=4,
            zero_division=0,
        )
    )
    print(
        "Precision measures how many predicted positive reviews were truly positive. "
        "Recall measures how many truly positive reviews were found. A threshold "
        "change can improve one measure while hurting the other, so the best "
        "balance depends on the application."
    )


if __name__ == "__main__":
    main()
