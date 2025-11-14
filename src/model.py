"""
CNN model definition for Natural Disaster Damage Detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def build_cnn(input_shape=(200, 200, 3), num_classes=6, learning_rate=2e-4):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                          input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(2048, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
