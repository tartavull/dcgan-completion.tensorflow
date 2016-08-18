from single_model_32 import Model
from glob import glob
import tensorflow as tf
from queue_runner import QueueRunner

filename_list = list(glob('/usr/people/it2/seungmount/research/datasets/mnist_png/training/all/*.png'))
assert filename_list
with tf.device("/gpu:0"):
  QueueRunner(filename_list, Model, image_size=32, image_channels=1)
