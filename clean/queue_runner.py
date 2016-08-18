import tensorflow as tf
import numpy as np
from datetime import datetime
from tqdm import tqdm

class QueueRunner(object):
  def __init__(self, filename_list, Model, batch_size=32, image_size=256, image_channels=3):
    #We create a tf variable to hold the global step, this has the effect
    #that when a checkpoint is created this value is saved.
    #Making the plots in tensorboard being continued when the model is restored.
    global_step = tf.Variable(0)
    increment_step = global_step.assign_add(1)

    #Create a queue that will be automatically fill by another thread
    #as we read batches out of it
    batch = self.batch_queue(filename_list, batch_size, image_size, image_channels)

    # Create the graph, etc.
    m = Model(batch)

    init_op = tf.initialize_all_variables()
    #This is required to intialize num_epochs for the filename_queue
    init_local = tf.initialize_local_variables()

    # Create a saver.
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    # Create a session for running operations in the Graph.
    sess = tf.Session(config=tf.ConfigProto(
      gpu_options = tf.GPUOptions(allow_growth=True),
      log_device_placement=False,
       allow_soft_placement=True))

    # Create a summary writer, add the 'graph' to the event file.
    log_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = tf.train.SummaryWriter('./logs/'+log_datetime,
      sess.graph, flush_secs=30, max_queue=2)

    # Initialize the variables (like the epoch counter).
    sess.run([init_op,init_local])

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:

      progress = tqdm()
      while not coord.should_stop():
          # Run training steps or whatever
          global_step = sess.run(increment_step)
          progress.update()
          m.step(sess)

          if global_step % 10 == 0:
            m.summarize(sess, writer, global_step)

          if global_step % 2000 == 0:
            # Append the step number to the checkpoint name:
            saver.save(sess, './logs/'+log_datetime, global_step=global_step)
            
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


  def read_png(self,filename_queue, image_size=256, image_channels=3):
    reader = tf.WholeFileReader()
    _ , value = reader.read(filename_queue)
    raw_int_image = tf.image.decode_png(value, channels=image_channels)
    center_cropped_image = tf.image.resize_image_with_crop_or_pad(raw_int_image, image_size, image_size)
    float_image = tf.cast(center_cropped_image,tf.float32)
    float_image = tf.sub(tf.div(float_image, 127.5), 1.0)

    #required for graph shape inference
    float_image.set_shape((image_size,image_size,image_channels))

    return float_image

  def batch_queue(self,filenames, batch_size, image_size, image_channels, num_epochs=None):

    with tf.variable_scope("batch_queue"):
      filename_queue = tf.train.string_input_producer(
          filenames, num_epochs=num_epochs, shuffle=True)
      image_node = self.read_png(filename_queue, image_size, image_channels)
      # min_after_dequeue defines how big a buffer we will randomly sample
      #   from -- bigger means better shuffling but slower start up and more
      #   memory used.
      # capacity must be larger than min_after_dequeue and the amount larger
      #   determines the maximum we will prefetch.  Recommendation:
      #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
      min_after_dequeue = 1000
      capacity = min_after_dequeue + 3 * batch_size
      example_batch = tf.train.shuffle_batch(
          [image_node], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
      return example_batch

class QueueRunner3D(QueueRunner):

  def read_png(self,filename_queue, image_size=32, image_channels=1):
    reader = tf.WholeFileReader()
    _ , value = reader.read(filename_queue)
    raw_int_image = tf.image.decode_png(value, channels=image_channels)
    center_cropped_image = tf.image.resize_image_with_crop_or_pad(raw_int_image, image_size * image_size, image_size)
    float_image = tf.cast(center_cropped_image,tf.float32)
    float_image = tf.sub(tf.div(float_image, 127.5), 1.0)
    float_image = tf.reshape(float_image, shape=(32,32,32))

    #required for graph shape inference
    float_image.set_shape((image_size,image_size,image_size))

    return float_image
