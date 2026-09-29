"""
model.py
--------
CNN architectures for super-resolution. THREE options, selectable via
--model in train.py:

  "srcnn"       - classic SRCNN (Dong et al.), 3 conv layers. Same-size
                  refinement: input and output are the same pixel
                  dimensions (refines a bicubic-upsampled blurry image).

  "deep_resnet" - deeper residual CNN (EDSR-inspired). Same-size
                  refinement, more capacity than srcnn.

  "espcn"       - TRUE upsampling architecture (Shi et al., "Efficient
                  Sub-Pixel CNN"). Input is SMALLER than output - this is
                  what actually changes physical resolution (e.g. 10m
                  input -> genuinely more pixels covering the same
                  ground, not just a refined same-size image). Use this
                  when you want a real, selectable scale factor (2x, 3x,
                  4x, 5x...) mapping to a specific target resolution
                  (10m / scale = result, e.g. scale 4 -> 2.5m).

ALL THREE MODELS ARE FULLY CONVOLUTIONAL (Input shape (None, None, 3)) -
meaning a trained model can run on an image of ANY size at inference
time, not just the fixed patch size it was trained on. This is what lets
the web app accept an uploaded image's natural aspect ratio instead of
forcing it into a square.

DROPOUT: srcnn and deep_resnet now include real Dropout layers, so the
Monte-Carlo-Dropout uncertainty mapping in evaluate.py/app.py produces
genuine, non-trivial uncertainty estimates (previously these architectures
had no Dropout, so uncertainty maps were always near-zero - a known,
now-fixed limitation).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------------------------------------------------
# Same-size refinement models (srcnn, deep_resnet)
# ---------------------------------------------------------------------

def build_srcnn(input_shape=(None, None, 3)):
    """Classic SRCNN with Dropout added for genuine uncertainty estimation."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (9, 9), padding="same", activation="relu", name="patch_extraction")(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(32, (1, 1), padding="same", activation="relu", name="nonlinear_mapping")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Conv2D(input_shape[-1] or 3, (5, 5), padding="same", activation="sigmoid", name="reconstruction")(x)

    return models.Model(inputs, outputs, name="SRCNN")


def residual_block(x, filters=64, dropout_rate=0.1):
    skip = x
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)
    return x


def build_deep_resnet(input_shape=(None, None, 3), num_blocks=8, filters=64):
    """Deeper residual CNN with Dropout added for genuine uncertainty estimation."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(inputs)
    initial_features = x

    for _ in range(num_blocks):
        x = residual_block(x, filters)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.Add()([x, initial_features])

    outputs = layers.Conv2D(input_shape[-1] or 3, (3, 3), padding="same", activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="DeepResNet_SR")


# ---------------------------------------------------------------------
# TRUE upsampling model: ESPCN (sub-pixel convolution)
# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="srm")
class PixelShuffle(layers.Layer):
    """
    Rearranges scale*scale*channels feature maps into a single image
    scale times larger in height and width ("sub-pixel convolution").

    This MUST be a proper registered Layer, not a Lambda wrapping a Python
    function - Keras blocks deserializing Lambda layers with arbitrary
    Python code when loading a saved model in a fresh process (a real
    failure mode this project hit and fixed: the model trained and saved
    fine in one session, but evaluate.py/app.py loading it afterward would
    have crashed with a "disallowed Lambda deserialization" error).
    """
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def build_espcn(input_shape=(None, None, 3), scale=4, filters=64):
    """
    Efficient Sub-Pixel CNN (Shi et al., 2016) - a real, published
    upsampling architecture. Unlike srcnn/deep_resnet, this model's OUTPUT
    is `scale` times larger than its input in both height and width -
    this is genuine super-resolution (more pixels out than in), not
    same-size refinement.

    How it works: feature extraction happens at the small LOW-resolution
    size (cheap), then the final layer produces scale*scale*channels
    feature maps, which get rearranged ("pixel shuffle" / depth-to-space)
    into a single image at the full HIGH-resolution size. This is more
    efficient and generally sharper than upsampling first and refining
    after (which is what srcnn/deep_resnet do).
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters, (5, 5), padding="same", activation="relu")(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters // 2, (3, 3), padding="same", activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters // 2, (3, 3), padding="same", activation="relu")(x)

    channels = input_shape[-1] or 3
    x = layers.Conv2D(channels * (scale ** 2), (3, 3), padding="same")(x)
    outputs = PixelShuffle(scale, name=f"pixel_shuffle_x{scale}")(x)
    outputs = layers.Activation("sigmoid")(outputs)

    return models.Model(inputs, outputs, name=f"ESPCN_x{scale}")


def build_model(model_type="srcnn", input_shape=(None, None, 3), scale=4):
    if model_type == "srcnn":
        return build_srcnn(input_shape)
    elif model_type == "deep_resnet":
        return build_deep_resnet(input_shape)
    elif model_type == "espcn":
        return build_espcn(input_shape, scale=scale)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'srcnn', 'deep_resnet', or 'espcn'.")


if __name__ == "__main__":
    # Quick sanity check when run directly
    for name in ["srcnn", "deep_resnet"]:
        m = build_model(name, input_shape=(64, 64, 3))
        print(f"\n{name} - {m.count_params():,} parameters")

    for scale in [2, 4]:
        m = build_model("espcn", input_shape=(16, 16, 3), scale=scale)
        print(f"\nespcn x{scale} - {m.count_params():,} parameters")
        print(f"  Input shape: {m.input_shape} -> Output shape: {m.output_shape}")
