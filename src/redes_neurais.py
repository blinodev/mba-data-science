# urc/redes_neurais.py

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

class TransformerRegressor(tf.keras.Model):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = layers.Dense(d_model)
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            for _ in range(num_layers)
        ]
        self.ffn_layers = [
            models.Sequential([
                layers.Dense(dim_feedforward, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(d_model),
                layers.Dropout(dropout),
            ]) for _ in range(num_layers)
        ]
        self.layernorms1 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.layernorms2 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)
        self.final_dense = layers.Dense(1)

    def call(self, x, training=False):
        # x shape: (batch_size, features)
        x = self.input_proj(x)  # (batch_size, d_model)
        x = tf.expand_dims(x, axis=1)  # (batch_size, seq_len=1, d_model)

        for mha, ffn, ln1, ln2 in zip(self.encoder_layers, self.ffn_layers, self.layernorms1, self.layernorms2):
            attn_output = mha(x, x, x, training=training)
            out1 = ln1(x + attn_output)
            ffn_output = ffn(out1, training=training)
            x = ln2(out1 + ffn_output)

        x = tf.squeeze(x, axis=1)  # (batch_size, d_model)
        out = self.final_dense(x)  # (batch_size, 1)
        return tf.squeeze(out, axis=1)  # (batch_size,)



import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def prever_transformer(model, X):
    preds = model.predict(X)
    return preds.squeeze()




