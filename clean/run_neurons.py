from model_neurons import ModelNeurons
from glob import glob
import tensorflow as tf
from queue_runner import QueueRunner3D

filename_list = list(glob('/root/seungmount/research/datasets/neuron_png/*.png'))
assert filename_list
with tf.device("/gpu:0"):
  QueueRunner3D(filename_list, ModelNeurons, image_size=32, image_channels=1)
