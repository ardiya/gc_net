from dataloader import SceneFlowDataset, SceneFlowLoader, PrepData
from gcnet import GCNet
from util import CustomProgBarLogger, allow_gpu_memory_growth, TensorboardImageCallback

from os.path import exists
from sys import exit

import argparse
import tensorflow as tf
import yaml

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train GC Net")
    parser.add_argument(
        "--config", default="", type=str, help="path to the yaml config file"
    )
    parser.add_argument("--allow-gpu-memory-growth", default=True)
    args = parser.parse_args()

    # Validate inputs
    if args.config == "":
        print("[Error]: --config must be specified")
        exit(1)
    elif not exists(args.config):
        print("[Error]: '%s' file doesn't exist" % args.config)
        exit(1)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg_dataset_train = cfg["dataset"]["train"]
        cfg_dataset_val = cfg["dataset"]["val"]
        cfg_net = cfg["net"]
        cfg_model = cfg["model"]

    if args.allow_gpu_memory_growth:
        allow_gpu_memory_growth()

    # Image Preprocessor
    prep_data = PrepData(
        cfg_net["im_height"], cfg_net["im_width"], cfg_net["max_disparity"]
    )

    # Training dataset
    train_loader = SceneFlowLoader(
        cfg_dataset_train["path_to_left"],
        cfg_dataset_train["path_to_right"],
        cfg_dataset_train["path_to_disparity"],
        prep_data=prep_data,
    )
    train_dataset = SceneFlowDataset(train_loader)
    train_dataset = train_dataset.batch(cfg_model["batch_size"])

    # Training Callbacks
    tb_callback = TensorboardImageCallback(
        log_dir=cfg_model["log_dir"], max_disparity=cfg_net["max_disparity"]
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cfg_model["checkpoint_dir"], save_weights_only=True, verbose=1
    )
    pb_callback = CustomProgBarLogger(steps_per_epoch=len(train_loader))

    # Model
    net = GCNet(cfg_net["max_disparity"])
    net.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.Huber(),
        run_eagerly=True,
    )
    net.fit(
        train_dataset,
        callbacks=[tb_callback, cp_callback, pb_callback],
        epochs=cfg_model["max_epoch"],
    )
