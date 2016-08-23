import tensorflow as tf
from glob import glob
import numpy as np

def run():

  batch = input_pipeline(list(glob('/usr/people/it2/seungmount/research/datasets/lsum_png/*.png')),10)
  # Create the graph, etc.
  init_op = tf.initialize_all_variables()

  # Create a session for running operations in the Graph.
  sess = tf.Session()

  # Initialize the variables (like the epoch counter).
  sess.run(init_op)

  # Start input enqueue threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  try:
      while not coord.should_stop():
          # Run training steps or whatever
          images = sess.run(batch)
          print np.array(images).shape

  except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
  finally:
      # When done, ask the threads to stop.
      coord.request_stop()

  # Wait for threads to finish.
  coord.join(threads)
  sess.close()


def read_my_file_format(filename_queue):
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  my_image = tf.image.decode_png(value)
  my_image_float = tf.cast(my_image,tf.float32)
  image_mean = tf.reduce_mean(my_image_float)
  my_noise = tf.random_normal([256,256,3],mean=image_mean)
  my_image_noisy = my_image_float + my_noise
  return my_image_noisy

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  image_node = read_my_file_format(filename_queue)
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

####################
if __name__ == '__main__':
  run()