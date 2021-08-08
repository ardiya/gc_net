from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Conv3D
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv3DTranspose
from tensorflow.python.ops.init_ops_v2 import he_normal
from softargmin import SoftArgMin


class _ConvBR_2D(Layer):
    """
    Conv2D BN ReLU
    """

    def __init__(self, n_feature, kernel_size, strides=(1, 1)):
        super(_ConvBR_2D, self).__init__()
        self.conv = Conv2D(
            n_feature,
            kernel_size,
            strides,
            padding="same",
            # kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
        self.bn = BatchNormalization()

    @tf.function
    def call(self, x, is_training):
        x = self.conv(x)
        x = self.bn(x, training=is_training)
        x = tf.nn.relu(x)

        return x


class _ConvBR_3D(Layer):
    """
    Conv3D BN ReLU
    """

    def __init__(self, n_feature, kernel_size, strides=(1, 1, 1)):
        super(_ConvBR_3D, self).__init__()
        self.conv = Conv3D(
            n_feature,
            kernel_size,
            strides,
            padding="same",
            # kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
        self.bn = BatchNormalization()

    @tf.function
    def call(self, x, is_training):
        x = self.conv(x)
        x = self.bn(x, training=is_training)
        x = tf.nn.relu(x)

        return x


class _DeconvBR_3D(Layer):
    """
    DeConv3D BN ReLU
    """

    def __init__(self, n_feature, kernel_size, strides=(2, 2, 2)):
        super(_DeconvBR_3D, self).__init__()
        self.conv = Conv3DTranspose(
            n_feature,
            kernel_size,
            strides,
            padding="same",
            # kernel_initializer=tf.keras.initializers.he_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        )
        self.bn = BatchNormalization()

    @tf.function
    def call(self, x, is_training):
        x = self.conv(x)
        x = self.bn(x, training=is_training)
        x = tf.nn.relu(x)

        return x


class _GCNetUnary(Layer):
    """
    Unary part (Section 3.1) of GCNet paper
    """

    def __init__(self):
        super(_GCNetUnary, self).__init__()
        self.conv1 = _ConvBR_2D(32, 5, strides=(2, 2))

        self.conv_a = list()
        self.conv_b = list()
        for _ in range(7):
            self.conv_a.append(_ConvBR_2D(32, 5))
            self.conv_b.append(_ConvBR_2D(32, 5))

        self.conv_final = Conv2D(32, 3, padding="same")

    @tf.function
    def call(self, x, is_training):
        x = self.conv1(x, is_training)

        for i in range(7):
            residual = x
            x = self.conv_a[i](x, is_training)
            x = self.conv_b[i](x, is_training)
            x = x + residual

        x = self.conv_final(x)

        return x


class _GCNetCostVolume(Layer):
    """
    Cost Volume part (Section 3.2) of GCNet paper
    """

    def __init__(self, n_disparity):
        super(_GCNetCostVolume, self).__init__()
        assert n_disparity % 2 == 0
        self.n_disparity = n_disparity

    @tf.function
    def call(self, left, right):
        # [N, H, W, C]  -> [N, 1, H, W, C]
        left = tf.expand_dims(left, axis=1)
        right = tf.expand_dims(right, axis=1)
        W = right.shape[3]
        assert self.n_disparity // 2 < W  # Disparity must be lower than W
        out = list()
        for d in range(self.n_disparity // 2):
            right_shifted = self._pad_left(right[:, :, :, : W - d, :], d)
            left_right_combined = tf.concat([left, right_shifted], axis=4)
            out.append(left_right_combined)
        # [N, n_disparity, W, H, C]
        out = tf.concat(out, axis=1)
        return out

    def _pad_left(self, x, left_val):
        return tf.pad(x, [[0, 0], [0, 0], [0, 0], [left_val, 0], [0, 0]])


class _GCNetRegularization(Layer):
    """
    Regularization part (Section 3.3) of GCNet Paper
    """

    def __init__(self):
        super(_GCNetRegularization, self).__init__()
        self.conv1 = _ConvBR_3D(32, 3)
        self.conv2 = _ConvBR_3D(32, 3)

        self.conv3 = _ConvBR_3D(64, 3, strides=(2, 2, 2))
        self.conv4 = _ConvBR_3D(64, 3)
        self.conv5 = _ConvBR_3D(64, 3)

        self.conv6 = _ConvBR_3D(64, 3, strides=(2, 2, 2))
        self.conv7 = _ConvBR_3D(64, 3)
        self.conv8 = _ConvBR_3D(64, 3)

        self.conv9 = _ConvBR_3D(64, 3, strides=(2, 2, 2))
        self.conv10 = _ConvBR_3D(64, 3)
        self.conv11 = _ConvBR_3D(64, 3)

        self.conv12 = _ConvBR_3D(128, 3, strides=(2, 2, 2))
        self.conv13 = _ConvBR_3D(128, 3)
        self.conv14 = _ConvBR_3D(128, 3)

        self.deconv1 = _DeconvBR_3D(64, 3, strides=(2, 2, 2))
        self.deconv2 = _DeconvBR_3D(64, 3, strides=(2, 2, 2))
        self.deconv3 = _DeconvBR_3D(64, 3, strides=(2, 2, 2))
        self.deconv4 = _DeconvBR_3D(32, 3, strides=(2, 2, 2))
        self.deconv_final = Conv3DTranspose(1, 3, strides=(2, 2, 2), padding="same")

    @tf.function
    def call(self, cost_volume, is_training):
        conv1 = self.conv1(cost_volume, is_training)
        conv2 = self.conv2(conv1, is_training)

        conv3 = self.conv3(cost_volume, is_training)
        conv4 = self.conv4(conv3, is_training)
        conv5 = self.conv5(conv4, is_training)

        conv6 = self.conv6(conv3, is_training)
        conv7 = self.conv7(conv6, is_training)
        conv8 = self.conv8(conv7, is_training)

        conv9 = self.conv9(conv6, is_training)
        conv10 = self.conv10(conv9, is_training)
        conv11 = self.conv11(conv10, is_training)

        conv12 = self.conv12(conv9, is_training)
        conv13 = self.conv13(conv12, is_training)
        conv14 = self.conv14(conv13, is_training)

        deconv1 = self.deconv1(conv14, is_training)
        deconv1 = deconv1 + conv11

        deconv2 = self.deconv2(deconv1, is_training)
        deconv2 = deconv2 + conv8

        deconv3 = self.deconv3(deconv2, is_training)
        deconv3 = deconv3 + conv5

        deconv4 = self.deconv4(deconv3, is_training)
        deconv4 = deconv4 + conv2

        deconv_final = self.deconv_final(deconv4)

        # [N, D, H, W, 1] -> [N, D, H, W]
        return tf.squeeze(deconv_final, axis=-1)


class GCNet(Model):
    """
    End-to-End Learning of Geometry and Context for Deep Stereo Regression
    https://arxiv.org/abs/1703.04309
    """

    def __init__(self, max_disparity):
        super(GCNet, self).__init__()
        self._max_disparity = max_disparity
        self.unary_block = _GCNetUnary()
        self.cost_volume_block = _GCNetCostVolume(max_disparity)
        self.regularization_block = _GCNetRegularization()
        self.soft_argmin = SoftArgMin(max_disparity)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, data, training=False):
        left, right = data
        left = self.unary_block(left, training)
        right = self.unary_block(right, training)

        cost_volume = self.cost_volume_block(left, right)
        reg = self.regularization_block(cost_volume, training)
        disparity = self.soft_argmin(reg)

        return disparity

    @tf.function
    def train_step(self, data):
        left, right, gt = data

        with tf.GradientTape() as tape:
            pred = self((left, right), training=True)
            loss_val = self.loss(pred, gt)

        grads = tape.gradient(loss_val, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss_val)

        lr = float(self.optimizer.get_config()["learning_rate"])

        return {
            "loss": self.loss_tracker.result(),
            "lr": tf.constant(lr),
            "left": left,
            "gt": gt,
            "pred": pred,
        }

    @property
    def metrics(self):
        return [self.loss_tracker]
