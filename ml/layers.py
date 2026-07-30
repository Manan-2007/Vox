"""Custom Keras layers shared by training and inference.

Like ml/normalize.py, this exists because the same definition has to be
available on both sides. `ml/train.py` builds the model with these layers and
`backend/main.py` loads the saved model, which can only reconstruct a custom
layer if the class is importable and registered. Defining it in train.py would
mean the backend could not load its own model.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="vox")
class AttentionPooling(keras.layers.Layer):
    """Collapse a sequence to one vector with learned per-timestep weights.

    The obvious alternative — take the final timestep — is a poor fit for a
    sliding window over live video. The window is not aligned to the sign: it
    may land halfway through, or after it has finished, so the last frame is
    frequently the least informative one in the window. Learning which frames
    carry the sign removes that assumption.

    Masking is handled explicitly rather than left to Keras. Composing
    `Dense(1)` with a stock `Softmax` layer looks like it should work and does
    not: the mask arrives shaped (batch, time) while the scores are
    (batch, time, 1), and Keras refuses to broadcast one to the other. Adding a
    large negative number to masked positions before the softmax is both the
    conventional fix and clearer about what it is doing — padded frames get
    weight zero, and the remaining weights still sum to one.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.score = keras.layers.Dense(1, activation="tanh", name="score")

    def build(self, input_shape):
        self.score.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        scores = self.score(inputs)  # (batch, time, 1)
        if mask is not None:
            keep = ops.expand_dims(ops.cast(mask, scores.dtype), axis=-1)
            # -1e9 rather than -inf: infinities produce NaNs if a whole sequence
            # is masked, which happens when the signer briefly leaves frame.
            scores = scores + (1.0 - keep) * -1e9
        weights = ops.softmax(scores, axis=1)
        return ops.sum(inputs * weights, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        # The time axis is gone, so any mask over it is meaningless downstream.
        return None
