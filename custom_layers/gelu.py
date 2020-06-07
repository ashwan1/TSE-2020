import math

import tensorflow as tf


class GELU(tf.keras.layers.Layer):
    """Code taken from tensorflow addons. https://github.com/tensorflow/addons/
    """

    def __init__(self, approximate: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.approximate = approximate
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return self.gelu(inputs)

    @tf.function
    def gelu(self, x):
        x = tf.convert_to_tensor(x)
        if self.approximate:
            pi = tf.cast(math.pi, x.dtype)
            coeff = tf.cast(0.044715, x.dtype)
            return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))
        else:
            return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))

    def get_config(self):
        config = {"approximate": self.approximate}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape
