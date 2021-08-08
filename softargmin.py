import tensorflow as tf
from tensorflow.keras.layers import Layer


class SoftArgMin(Layer):
    def __init__(self, n_disparity):
        super(SoftArgMin, self).__init__()
        self.disparity_indices = tf.range(n_disparity, dtype=tf.float32)
        self.disparity_indices = tf.reshape(self.disparity_indices, [1, n_disparity, 1, 1])

    def call(self, x):
        # [N, D, H, W]
        x = tf.nn.softmax(x, axis=1)  # compute softmax over all disparity
        x = tf.math.multiply(x, self.disparity_indices)
        # [N, D, H, W] -> [N, H, W]
        x = tf.math.reduce_sum(x, axis=1)
        
        return x
