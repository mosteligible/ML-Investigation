from pathlib import Path
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model


PATH = Path.cwd().parent.absolute()

with open(PATH / "test_models/feature.log", "r") as fp:
    FEATURES = fp.read().split(" ")
MODEL = load_model(PATH / "/server/model-weights.h5")
FEATURES_SIZE = len(FEATURES)


def predict(payload):
    input_text = payload.lower()
    print("Input splitted payload", input_text.split())
    input_text = Counter(input_text.split())
    print("\nCounter of input split text:", input_text)
    input_feature = [0] * FEATURES_SIZE  # np.zeros(shape=(1, FEATURES_SIZE))
    for word, occurence in input_text.items():
        try:
            word_index = FEATURES.index(word)
        except ValueError:
            print(f"{word} does not exist in vocabulary!")
            continue
        input_feature[word_index] = occurence
    input_feature = [input_feature, ]
    prediction = MODEL.predict(input_feature)
    return prediction


if __name__ == "__main__":
    x = predict("Since the topic of the sentence is completely unknown it forces the writer to be creative when the sentence appears There are a number of different ways a writer can use the random sentence for creativity The most common way to use the sentence is to begin a story")
    print("\n\nPrediction", x[0, 0], type(x), x.dtype)
