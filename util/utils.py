import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import tensorflow as tf
from tensorflow import keras
from zipfile import ZipFile

def log2(x):
    return int(np.log2(x))

def plot_images(images, log2_res, name_file = ""):
    scales = {2: 0.5, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
    scale = scales[log2_res]

    grid_col = min(images.shape[0], int(32 // scale))
    grid_row = 1

    figure, axes = plt.subplots(grid_row, grid_col, figsize = (grid_col * scale, grid_row * scale))
    for row in range(grid_row):
        ax = axes if grid_row == 1 else axes[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col])
            ax[col].axis("off")
    plt.show()
    figure
    if name_file:
        figure.savefig(name_file)
def fade_in(alpha, a, b):
    return alpha *a + (1 - alpha) * b
def wasserstein_loss(y_true, y_pred): # # tf.reduce_mean(y_true * y_pred) xấp xỉ Wasserstein distance theo một cách cụ thể. 
    return -tf.reduce_mean(y_true * y_pred)
def pixel_norm(x, e = 1e-8):
    return x / tf.math.sqrt(tf.reduce_mean(x**2))
def minibatch_std(input_tensor, e = 1e-8):
    n_sample, h, w, channel = tf.shape(input_tensor)
    group_size = tf.minimum(4,n_sample)
    x = tf.reshape(input_tensor, [group_size, -1, h, w, channel])
    group_mean, group_var = tf.nn.moments(x, axes =[0], keepdims = False) # tính toán phương sai và độ lệch chuẩn
    group_std = tf.sqrt(group_var + e)
    avg_std = tf.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True) # Giá trị trung bình của group_std trên các trục 
    x = tf.tile(avg_std, [group_size, h, w, 1]) # sao chép avg_std thành shape mới 
    return tf.concat([input_tensor, x], axis=-1)

