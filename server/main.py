from fastapi import FastAPI
from pydantic import BaseModel
from uvicorn import run

from collections import Counter
from pathlib import Path
import tensorflow as tf
import numpy as np


app = FastAPI()
PATH = Path.cwd()
test_model_dir = PATH.parent / "test_models"
model_path = PATH / "model-weights.h5"
MODEL = tf.keras.models.load_model(model_path)
with open(test_model_dir / "feature.log", "r") as fp:
    FEATURES = fp.read().split(" ")
FEATURES_SIZE = len(FEATURES)


class Item(BaseModel):
    Payload: str


@app.post("/predict/")
def predict(payload: Item):
    input_text = payload.Payload.lower()
    input_text = Counter(input_text.split())
    input_feature = np.zeros(shape=(1, FEATURES_SIZE), dtype=int)
    for word, occurence in input_text.items():
        try:
            word_index = FEATURES.index(word)
        except ValueError:
            # Lazy feature build, definitely not the right way because we are losing the information
            print(f"{word} does not exist in vocabulary!")
            continue
        input_feature[0, word_index] = occurence

    prediction = MODEL.predict(input_feature)
    return {"soft-weight": float(prediction[0, 0])}


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
