import tensorflow as tf
from glob import glob
import numpy as np
from datetime import datetime

def run(filename_list, batch_size, Model):

  #Create a queue that will be automatically fill by another thread
  #as we read batches out of it
  global_step = tf.Variable(0)
  increment = global_step.assign_add(1)
  batch = batch_queue(filename_list, batch_size)

  # Create the graph, etc.
  m = Model(batch_size)

  init_op = tf.initialize_all_variables()
  #This is required to intialize num_epochs for the filename_queue
  init_local = tf.initialize_local_variables()

  # Create a session for running operations in the Graph.
  sess = tf.Session(config=tf.ConfigProto(
    log_device_placement=False))

  # Create a summary writer, add the 'graph' to the event file.
  log_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  writer = tf.train.SummaryWriter('./logs/'+log_datetime, sess.graph)

  # Initialize the variables (like the epoch counter).
  sess.run([init_op,init_local])

  # Start input enqueue threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  try:

    while not coord.should_stop():
        # Run training steps or whatever
        global_step, _, summary = sess.run([increment, m.step(), m.summarize(writer)])
        writer.add_summary(summary, global_step=global_step)
        # print np.array(z).shape



  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()

  # Wait for threads to finish.
  coord.join(threads)
  sess.close()


def read_png(filename_queue, image_size=256, image_channels=3):
  reader = tf.WholeFileReader()
  _ , value = reader.read(filename_queue)
  raw_int_image = tf.image.decode_png(value, channels=image_channels)
  center_cropped_image = tf.image.resize_image_with_crop_or_pad(raw_int_image, image_size, image_size)
  float_image = tf.cast(center_cropped_image,tf.float32)

  #required for graph shape inference
  float_image.set_shape((image_size,image_size,image_channels))

  return float_image

def batch_queue(filenames, batch_size, num_epochs=1):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  image_node = read_png(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 100
  capacity = min_after_dequeue + 3 * batch_size
  example_batch = tf.train.shuffle_batch(
      [image_node], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch


class Model(object):

  def __init__(self, batch_size, z_size=100):
    with tf.variable_scope("model"):

      self.z = tf.random_uniform(shape=(batch_size, z_size), 
                            minval=-1.0, maxval=1.0, 
                            dtype=tf.float32, seed=None, name='z')
      self._step = 0
      self.z_sum = tf.histogram_summary("z_hist", self.z)
  def step(self):
    return self.z

  def summarize(self, writer):
    return self.z_sum

if __name__ == '__main__':
  filename_list = list(glob('/usr/people/it2/seungmount/research/datasets/lsum_png_copy/*.png'))
  run(filename_list,10, Model)