# positional_embedding.py
import tensorflow as tf
from keras.utils import register_keras_serializable

@register_keras_serializable()
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embedding_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = self.pos_emb(positions)
        return x + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim
        })
        return config
