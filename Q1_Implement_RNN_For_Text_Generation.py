#!/usr/bin/env python3
"""Train a small LSTM character model and generate text."""


# imports and setup code are provided for you. You may add additional imports if you wish.
import argparse
import os
import random
import urllib.request # for downloading the Tiny Shakespeare dataset

import numpy as np
import tensorflow as tf


# This is a short backup text, so the example can still run without internet.
FALLBACK_TEXT = """
From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thy self thy foe, to thy sweet self too cruel:
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And, tender churl, mak'st waste in niggarding:
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
""" * 20

# Stores URL for data
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


# this function defines the text loading logic
def load_text(data_path):
    """Load a text file, or download Tiny Shakespeare if no file is given."""
    # The user can provide a local text file. This is useful when experimenting
    # with a different writing style instead of always using Shakespeare.
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as text_file:
            return text_file.read()

    # included a try catch for exception scenario so taht we can handle that gracefully and use the fallback text if the download fails.
    try:
        # Downloading a real text corpus gives the LSTM many examples of how
        # characters commonly appear next to one another.
        print("Downloading the Tiny Shakespeare dataset...")
        with urllib.request.urlopen(DATA_URL, timeout=15) as response:
            return response.read().decode("utf-8")
    except Exception as error:
        print(f"Dataset download failed ({error}). Using the small backup text.")
        # return buil in text if the download fails. This is a small sample of Shakespeare's writing.
        return FALLBACK_TEXT

# this function defines the training data creation logic
def make_training_data(text, sequence_length):
    """Turn characters into integer sequences for supervised learning."""
    # As a neural network cannot use letters directly, so we first create a small
    # dictionnary such as {'a': 0, 'b': 1, ...}. The dictionary is learned from
    # the actuall text, which means it also includes spaces and punctuation.
    # Every different character gets one integer number. This is easier for an
    # Embedding layer than giving the model a large one-hot vector by hand.

    # find all quique characters in the text and sort them to make a consistent mapping.
    characters = sorted(set(text))

    #creating dictionaries to map characters to integer IDs and vice versa. This is useful for converting the text into a format that can be fed into the neural network.
    char_to_id = {character: index for index, character in enumerate(characters)}

    # creates reverse mapping from integer IDs back to characters. This is useful for converting the model's output back into readable text after generation.
    id_to_char = np.array(characters)

    # convert the entire text into a sequence of integer IDs using the char_to_id mapping. This transforms the text into a numerical format suitable for training the LSTM model.
    encoded_text = np.array([char_to_id[character] for character in text])

    # We make many overlapping examples. For example, with sequence length 4,
    # "abcd" is used to predict "e", then "bcde" is used to predict "f".
    # The input is a group of characters and the target is the next character.
    sample_count = len(encoded_text) - sequence_length
    inputs = np.zeros((sample_count, sequence_length), dtype=np.int32)
    targets = np.zeros(sample_count, dtype=np.int32)
    for index in range(sample_count):
        inputs[index] = encoded_text[index : index + sequence_length]
        targets[index] = encoded_text[index + sequence_length]

    return inputs, targets, char_to_id, id_to_char


def build_model(vocabulary_size, embedding_size, lstm_units):
    """Create the LSTM RNN which predicts one character."""
    # The model receives character IDs and returns one score for every possible
    # character. The character with the highest score is the most likley next
    # character, although generation below samples from all scores.
    # Embedding changes character IDs into useful learned number vectors.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(None,)),
            tf.keras.layers.Embedding(vocabulary_size, embedding_size),
            # LSTM remembers information from earlier characters in the sequence.
            # It can learn patterns such as spaces after words and punctuashun
            # after a phrase.
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(vocabulary_size),
        ]
    )
    # Sparse loss accepts the integer target character directly.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        # Sparse categorical cross-entropy compares the correct character ID
        # with the scores produced by the final Dense layer.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def sample_next_character(logits, temperature):
    """Pick one character, with temperature controlling randomness."""
    # Logits are raw model scores. Dividing by temperature changes how sharply
    # the probabilities favor the best predicton: low values are conservitive,
    # while high values make unusual characters more likely.
    # A low temperature makes the largest prediction more likely. A high one
    # makes less likely characters have more chance to be selected.
    adjusted_logits = logits / max(temperature, 1e-5)
    probabilities = tf.nn.softmax(adjusted_logits).numpy()
    return np.random.choice(len(probabilities), p=probabilities)


def generate_text(
    model, start_text, char_to_id, id_to_char, sequence_length, length, temperature
):
    """Generate characters one by one from a starting prompt."""
    generated = start_text

    for _ in range(length):
        # At each step, the model reads the current context and predicts only
        # the next character. That new character is then added to the contex
        # and becomes part of the input for the following step.
        # Keep only the last characters because the model was trained on this size.
        recent_text = generated[-sequence_length:]
        input_ids = [char_to_id.get(character, 0) for character in recent_text]
        input_ids = tf.keras.utils.pad_sequences(
            [input_ids], maxlen=sequence_length, padding="pre"
        )
        logits = model(input_ids, training=False)[0]
        next_id = sample_next_character(logits, temperature)
        generated += str(id_to_char[next_id])

    return generated


def main():
    # These options let the student make the experiment smaller or larger.
    parser = argparse.ArgumentParser(description="Generate text with an LSTM RNN")
    parser.add_argument("--data-path", default="", help="Optional local text file")
    parser.add_argument(
        "--max-characters",
        type=int,
        default=0,
        help="Use only this many characters; 0 uses the complete dataset",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--generated-length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed-text", default="ROMEO:")
    args = parser.parse_args()

    # Setting seeds makes this example results more repeatable.
    random.seed(7)
    np.random.seed(7)
    tf.random.set_seed(7)

    text = load_text(args.data_path)
    if args.max_characters > 0:
        # A smaller text sample makes training much quicker for testing.
        text = text[: args.max_characters]
        print(f"Using only the first {len(text):,} characters for quicker training.")
    if len(text) <= args.sequence_length:
        raise ValueError("The text dataset must be longer than sequence length.")
    inputs, targets, char_to_id, id_to_char = make_training_data(
        text, args.sequence_length
    )
    print(f"Loaded {len(text):,} characters and {len(char_to_id)} unique characters.")

    dataset = (
        # Each input sequence is paired with the character that should come
        # immediately after it. Shuffling prevents the model from seeing all
        # examples in their original order during training.
        tf.data.Dataset.from_tensor_slices((inputs, targets))
        .shuffle(min(len(inputs), 10000), seed=7)
        .batch(args.batch_size)
    )
    model = build_model(len(char_to_id), embedding_size=64, lstm_units=128)
    model.summary()
    # During training, the model adjusts its weights to reduce prediction
    # mistakes. One epoch means the model has processed the training examples.
    model.fit(dataset, epochs=args.epochs)

    # Unknown prompt characters are replaced by the first known character.
    seed_text = "".join(
        character if character in char_to_id else " " for character in args.seed_text
    )
    print("\nGenerated text (temperature=" + str(args.temperature) + "):\n")
    print(
        generate_text(
            model,
            seed_text,
            char_to_id,
            id_to_char,
            args.sequence_length,
            args.generated_length,
            args.temperature,
        )
    )
    print(
        "\nTemperature note: lower values make text safer and more predictable; "
        "higher values make it more random and sometimes less sensible."
    )


if __name__ == "__main__":
    main()