import tensorflow as tf
from tensorflow.keras.losses import Loss


class MyHuberLoss(Loss):
    # initialize instance attributes
    def __init__(self, threshold=1):
        super(MyHuberLoss, self).__init__()
        self.threshold = threshold

    # Compute loss
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - self.threshold / 2)
        return tf.where(is_small_error, small_error_loss, big_error_loss)
