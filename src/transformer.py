from keras import layers, Input, Model

def dst_transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def dst_cria_modelo(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = layers.LSTM(10, return_sequences=True)(inputs)
    for _ in range(num_transformer_blocks):
        x = dst_transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GRU(100, return_sequences=False)(x)
    x = layers.Dropout(mlp_dropout)(x)
    x = layers.Dense(mlp_units, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    return Model(inputs, outputs)
