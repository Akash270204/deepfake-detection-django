import tensorflow as tf
from tensorflow import keras


def build_inference_model(trained_model):
    """
    Builds a clean inference-only model for Grad-CAM
    (no data augmentation layers)
    """

    # 🔹 Extract EfficientNet backbone
    backbone = trained_model.get_layer("efficientnetb1")
    backbone.trainable = True

    # 🔹 Create new input
    inputs = keras.Input(shape=(240, 240, 3))

    # 🔹 Forward pass (NO augmentation)
    x = backbone(inputs, training=False)

    # 🔹 Rebuild classifier layers (reuse weights)
    start_copy = False
    for layer in trained_model.layers:
        if layer.name == "efficientnetb1":
            start_copy = True
            continue

        if start_copy:
            x = layer(x)

    # 🔹 New inference model
    inference_model = keras.Model(inputs, x, name="inference_model")

    return inference_model
