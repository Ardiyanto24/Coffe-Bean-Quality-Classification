import tensorflow as tf
from .utils import freeze_backbone, get_classification_head


def build_mobilenetv2(
    input_shape,
    num_classes,
    weights="imagenet",
    freeze=True,
    dropout_rate=0.5,
    dense_units=256
):
    """
    Build MobileNetV2 model
    """

    inputs = tf.keras.Input(shape=input_shape)

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_tensor=inputs
    )

    if freeze:
        freeze_backbone(backbone)

    x = backbone.output
    outputs = get_classification_head(
        x,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")

    return model
