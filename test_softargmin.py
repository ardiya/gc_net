#!/usr/bin/env python3
"""
Scripts to test softargmin
"""
import tensorflow as tf
import numpy as np
from softargmin import SoftArgMin

n_batch = 8
im_height = 3
im_width = 4
n_disparity = 5

np_gt = np.random.randint(0, n_disparity, size=[n_batch, im_height, im_width])
np_data = np.random.uniform(
    low=0, high=1.0, size=[n_batch, n_disparity, im_height, im_width]
)
for b in range(n_batch):
    for r in range(im_height):
        for c in range(im_width):
            # get disparity value from gt
            disparity = np_gt[b, r, c]
            # assign big number to np_data[:,disparity,:,:] so that the softargmin result 
            # is an integer disparity value instead of value between 2 integer, though in practice we want to have 
            np_data[b, disparity, r, c] = 25.0

# create soft_argmin
soft_argmin = SoftArgMin(n_disparity)

# compare soft_argmin(data) to gt, it should be equal
gt = tf.constant(np_gt, dtype=tf.float32)
data = tf.constant(np_data, dtype=tf.float32)
soft_argmin_val = soft_argmin(data)
print("soft_argmin", soft_argmin_val.shape)
print("gt", gt.shape)
error = tf.reduce_sum(gt - soft_argmin_val)
print("error between softargmin and gt:", error.numpy())  # Error should be low
