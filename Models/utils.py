import tensorflow as tf


def freeze_backbone(backbone):
    """Freeze all layers in backbone"""
    for layer in backbone.layers:
        layer.trainable = False


def unfreeze_backbone(backbone, n_layers=None):
    """
    Unfreeze backbone.
    If n_layers is None -> unfreeze all
    Else -> unfreeze last n_layers
    """
    if n_layers is None:
        for layer in backbone.layers:
            layer.trainable = True
    else:
        for layer in backbone.layers[:-n_layers]:
            layer.trainable = False
        for layer in backbone.layers[-n_layers:]:
            layer.trainable = True


def get_classification_head(
    x,
    num_classes,
    dropout_rate=0.5,
    dense_units=256
):
    """
    Standard classification head
    """
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)
    return outputs
