from model_lsum import Model
from glob import glob
import tensorflow as tf
from queue_runner import QueueRunner

filename_list = list(glob('/root/seungmount/research/datasets/lsum_png/*.png'))
print len(filename_list)
with tf.device("/gpu:1"):
  QueueRunner(filename_list, Model, image_size=256, image_channels=3)
