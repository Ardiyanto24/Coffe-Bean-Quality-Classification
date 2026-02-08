import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

def preprocess_image_tf(image_path):
    """TensorFlow-based preprocessing"""
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize
    img = tf.image.resize(img, [300, 300], method='bilinear')
    
    # Convert to float32
    img = tf.cast(img, tf.float32)
    
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    
    # EfficientNet preprocessing
    img = preprocess_input(img)
    
    return img