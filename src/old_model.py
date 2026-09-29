"""
model.py
--------
CNN architectures for super-resolution. Two options, selectable via --model
in train.py:

  "srcnn"      - the original SRCNN architecture (Dong et al.). 3 conv layers.
                 Fast to train, good baseline, matches the "SRCNN" label in
                 your flowchart.

  "deep_resnet" - a deeper residual CNN (inspired by EDSR's residual-block
                 idea, simplified). More capacity, better detail recovery,
                 slower to train. Matches the "EDSR" direction in your
                 flowchart without the full complexity of the original paper.

Both models take a same-size input and output (the bicubic-upsampled LR image
in, the refined/sharpened image out) - see data_pipeline.py for why.
"""

from tensorflow.keras import layers, models


def build_srcnn(input_shape=(64, 64, 3)):
    """
    Classic SRCNN: patch extraction -> non-linear mapping -> reconstruction.
    ~57,000 parameters - trains quickly even on CPU.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (9, 9), padding="same", activation="relu", name="patch_extraction")(inputs)
    x = layers.Conv2D(32, (1, 1), padding="same", activation="relu", name="nonlinear_mapping")(x)
    outputs = layers.Conv2D(input_shape[-1], (5, 5), padding="same", activation="sigmoid", name="reconstruction")(x)

    model = models.Model(inputs, outputs, name="SRCNN")
    return model


def residual_block(x, filters=64):
    """One residual block: two conv layers with a skip connection.
    This is the core building block that makes deeper SR networks (EDSR-style)
    trainable without vanishing gradients."""
    skip = x
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.Add()([x, skip])
    x = layers.Activation("relu")(x)
    return x


def build_deep_resnet(input_shape=(64, 64, 3), num_blocks=8, filters=64):
    """
    Deeper residual CNN for higher-quality reconstruction, inspired by EDSR's
    residual-block design (simplified - no sub-pixel upsampling since our
    input/output are already the same size, per the SRCNN-style training setup).
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(inputs)
    initial_features = x

    for _ in range(num_blocks):
        x = residual_block(x, filters)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.Add()([x, initial_features])  # global residual connection

    outputs = layers.Conv2D(input_shape[-1], (3, 3), padding="same", activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="DeepResNet_SR")
    return model


def build_model(model_type="srcnn", input_shape=(64, 64, 3)):
    if model_type == "srcnn":
        return build_srcnn(input_shape)
    elif model_type == "deep_resnet":
        return build_deep_resnet(input_shape)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'srcnn' or 'deep_resnet'.")


if __name__ == "__main__":
    # Quick sanity check when run directly
    for name in ["srcnn", "deep_resnet"]:
        m = build_model(name)
        print(f"\n{name} — {m.count_params():,} parameters")
        m.summary()
