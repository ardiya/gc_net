import copy
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
from os.path import join

from tensorflow.python.keras.utils.generic_utils import Progbar


def allow_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def normalize_img_for_vis(img: np.ndarray):
    "Convert [-1,1] to [0, 255]"
    img = np.array((img + 1.0) / 2.0 * 255, dtype=np.uint8)

    return img


def normalize_disparity_for_vis(disp: np.ndarray, n_disparity):
    vis_img = np.clip(disp, 0, n_disparity) / n_disparity
    vis_img = np.array(vis_img * 255, dtype=np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

    return vis_img


class TensorboardImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, max_disparity):
        self.curr_batch = 0
        self.writer = tf.summary.create_file_writer(
            join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        )
        self.max_disparity = max_disparity

    def on_train_batch_end(self, _, logs):
        with self.writer.as_default():
            tf.summary.scalar("loss", logs["loss"], step=self.curr_batch)
            tf.summary.scalar("lr", logs["lr"], step=self.curr_batch)
            tf.summary.image("left", logs["left"], step=self.curr_batch)

            pred = self._process_disparity(logs["pred"])
            tf.summary.image("pred", pred, step=self.curr_batch)

            gt = self._process_disparity(logs["gt"])
            tf.summary.image("gt", gt, step=self.curr_batch)
        self.curr_batch += 1

    def _process_rgb(self, inputs):
        outs = list()
        for i in range(inputs.shape[0]):
            outs.append(
                cv2.cvtColor(normalize_img_for_vis(inputs[i, ...]), cv2.COLOR_BGR2RGB)
            )
        return np.array(outs, dtype=np.uint8)

    def _process_disparity(self, inputs):
        outs = list()
        for i in range(inputs.shape[0]):
            outs.append(
                cv2.cvtColor(
                    normalize_disparity_for_vis(inputs[i, ...], self.max_disparity),
                    cv2.COLOR_BGR2RGB,
                )
            )
        return np.array(outs, dtype=np.uint8)


class CustomProgBarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, steps_per_epoch, verbose=1):
        super(CustomProgBarLogger, self).__init__()
        self.seen = 0
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose

    def on_train_batch_end(self, _, logs=None):
        if self.progbar is None:
            self.progbar = Progbar(target=self.steps_per_epoch, verbose=self.verbose)

        # The only important thing is loss
        important_logs = {"loss": logs["loss"]}

        self.seen += 1
        self.progbar.update(self.seen, list(important_logs.items()), finalize=False)
