from typing import Tuple
from pfm_io import readPFM
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Layer
import tensorflow as tf
from pathlib import Path
import cv2
import numpy as np


class PrepData:
    """Preprocess Data"""

    def __init__(self, target_height, target_width, max_disparity):
        self.target_height = target_height
        self.target_width = target_width
        self.max_disparity = max_disparity

    def __call__(
        self, left: np.ndarray, right: np.ndarray, disparity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W, _ = left.shape

        rand = np.random.RandomState()
        r = rand.randint(0, H - self.target_height)
        c = rand.randint(0, W - self.target_width)

        # random cropping from [r,c] to [r+target_height, c+target_width]
        left = left[r : r + self.target_height, c : c + self.target_width, :]
        right = right[r : r + self.target_height, c : c + self.target_width, :]
        disparity = disparity[r : r + self.target_height, c : c + self.target_width]

        return (
            self._scale_img(left),
            self._scale_img(right),
            self._clamp_disparity(disparity),
        )

    def _scale_img(self, img):
        """Convert [0-255] to [0.0-1.0]"""
        return np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0

    def _clamp_disparity(self, disparity):
        """Clip max disparity, ortherwise it'll be hard for network to learn really big disparity/close object"""
        return np.clip(disparity, 0, self.max_disparity)


class SceneFlowLoader:
    def __init__(
        self, path_to_left, path_to_right, path_to_disparity, prep_data: PrepData
    ):
        """Populate all the png, png, pfm files in left, right, disparity folders respectively"""
        path_to_left = Path(path_to_left).resolve().expanduser()
        path_to_right = Path(path_to_right).resolve().expanduser()
        path_to_disparity = Path(path_to_disparity).resolve().expanduser()

        self.lefts = list()
        self.rights = list()
        self.disparities = list()
        self.prep_data = prep_data
        for left_filename in path_to_left.iterdir():
            name = left_filename.name
            if not name.endswith(".png"):
                continue
            right_filename = path_to_right / name
            disparity_filename = path_to_disparity / name.replace(".png", ".pfm")
            if not right_filename.exists():
                print("[Warning]:", right_filename, "doesn't exist")
            elif not disparity_filename.exists():
                print("[Warning]:", disparity_filename, "doesn't exist")
            else:
                self.lefts.append(left_filename)
                self.rights.append(right_filename)
                self.disparities.append(disparity_filename)

    def __len__(self):
        return len(self.lefts)

    def __getitem__(self, i):
        """Actual read file and preprocessing"""
        l = cv2.imread(str(self.lefts[i]))
        r = cv2.imread(str(self.rights[i]))
        d, _ = readPFM(str(self.disparities[i]))
        if d[0][0] < 0:  # make disparity positive if it's negative
            d *= -1

        return self.prep_data(l, r, d)


class SceneFlowDataset(tf.data.Dataset):
    def __new__(cls, loader: SceneFlowLoader):
        generator = lambda: (data for data in loader)
        return tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=(
                [loader.prep_data.target_height, loader.prep_data.target_width, 3],
                [loader.prep_data.target_height, loader.prep_data.target_width, 3],
                [loader.prep_data.target_height, loader.prep_data.target_width],
            ),
        )


if __name__ == "__main__":
    from util import normalize_disparity_for_vis, normalize_img_for_vis
    import tensorflow as tf

    img_height = 192
    img_width = 384
    max_disparity = 96
    prep_data = PrepData(img_height, img_width, max_disparity)

    loader = SceneFlowLoader(
        path_to_left="/home/ardiya/dataset/SceneFlow/train/image_clean/left",
        path_to_right="/home/ardiya/dataset/SceneFlow/train/image_clean/right",
        path_to_disparity="/home/ardiya/dataset/SceneFlow/train/disparity/left",
        prep_data=prep_data,
    )
    generator = lambda: (data for data in loader)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=(
            [img_height, img_width, 3],
            [img_height, img_width, 3],
            [img_height, img_width],
        ),
    )

    for left, right, gt in dataset.take(5):
        # visualize first 5
        cv2.imshow("prep_left", normalize_img_for_vis(left))
        cv2.imshow("prep_right", normalize_img_for_vis(right))
        cv2.imshow("prep_gt", normalize_disparity_for_vis(gt, max_disparity))

        cv2.waitKey(0)
