import tensorflow as tf
from .utils import freeze_backbone, get_classification_head


EFFICIENTNET_MAP = {
    "B0": tf.keras.applications.EfficientNetB0,
    "B1": tf.keras.applications.EfficientNetB1,
    "B2": tf.keras.applications.EfficientNetB2,
    "B3": tf.keras.applications.EfficientNetB3,
}


def build_efficientnet(
    variant,
    input_shape,
    num_classes,
    weights="imagenet",
    freeze=True,
    dropout_rate=0.5,
    dense_units=256
):
    """
    Build EfficientNet (B0–B3)
    """

    if variant not in EFFICIENTNET_MAP:
        raise ValueError(f"EfficientNet variant {variant} not supported")

    inputs = tf.keras.Input(shape=input_shape)

    backbone = EFFICIENTNET_MAP[variant](
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

    model = tf.keras.Model(
        inputs,
        outputs,
        name=f"EfficientNet{variant}"
    )

    return model
