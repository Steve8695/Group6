# utils.py
import numpy as np
import json
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from positional_embedding import PositionEmbedding

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_tokenizer(path="tokenizer.json"):
    with open(path, "r", encoding="utf-8") as f:
        return tokenizer_from_json(f.read())

def load_label_classes(path="label_classes.npy"):
    return np.load(path, allow_pickle=True)

def load_sentiment_model(path="Causal_Transformer.keras"):
    return load_model(path, compile=False)
    
def prepare_input(texts, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
