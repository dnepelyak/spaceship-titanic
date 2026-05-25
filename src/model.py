"""
Model architectures for Spaceship Titanic.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build_mlp(input_dim: int, params: dict) -> tf.keras.Model:
    """Standard deep MLP with residual-like skip connections."""
    reg = regularizers.l2(params.get("l2_reg", 1e-4))
    act = params.get("activation", "swish")
    units = params.get("units", 256)
    dropout = params.get("dropout", 0.3)
    n_layers = params.get("n_layers", 4)
    lr = params.get("lr", 1e-3)

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for i in range(n_layers):
        x = layers.Dense(units, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(act)(x)
        x = layers.Dropout(dropout)(x)

        # Halve units every 2 layers
        if (i + 1) % 2 == 0:
            units = max(units // 2, 32)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="mlp")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=params.get("l2_reg", 1e-4)),
        loss="binary_crossentropy",
        metrics=["AUC", "accuracy"],
    )
    return model


def build_wide_deep(input_dim: int, params: dict) -> tf.keras.Model:
    """Wide & Deep architecture — captures memorization + generalization."""
    reg = regularizers.l2(params.get("l2_reg", 1e-4))
    act = params.get("activation", "swish")
    units = params.get("units", 256)
    dropout = params.get("dropout", 0.3)
    lr = params.get("lr", 1e-3)

    inputs = tf.keras.Input(shape=(input_dim,))

    # Wide part (linear)
    wide = layers.Dense(1, kernel_regularizer=reg)(inputs)

    # Deep part
    x = inputs
    for d in [units, units // 2, units // 4]:
        x = layers.Dense(max(d, 32), kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(act)(x)
        x = layers.Dropout(dropout)(x)
    deep = layers.Dense(1, kernel_regularizer=reg)(x)

    combined = layers.Add()([wide, deep])
    outputs = layers.Activation("sigmoid")(combined)

    model = tf.keras.Model(inputs, outputs, name="wide_deep")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=params.get("l2_reg", 1e-4)),
        loss="binary_crossentropy",
        metrics=["AUC", "accuracy"],
    )
    return model
