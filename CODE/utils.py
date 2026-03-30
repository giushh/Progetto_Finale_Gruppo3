import keras
from keras import layers
import tensorflow as tf


@keras.saving.register_keras_serializable(package="Custom")
class ColorJitter(layers.Layer):
    def __init__(self, brightness=0.08, contrast=0.12, saturation=0.10, **kwargs):
        super().__init__(**kwargs)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def call(self, x, training=None):
        if training is False or training is None:
            return x
        x = tf.image.random_brightness(x, max_delta=self.brightness)
        x = tf.image.random_contrast(x, lower=1.0 - self.contrast, upper=1.0 + self.contrast)
        x = tf.image.random_saturation(x, lower=1.0 - self.saturation, upper=1.0 + self.saturation)
        return tf.clip_by_value(x, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
        })
        return config