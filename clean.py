import tensorflow as tf
from glob import glob
import numpy as np
from datetime import datetime
from ops import *
from utils import *
from tqdm import tqdm

def run(filename_list, batch_size, Model):
  #We create a tf variable to hold the global step, this has the effect
  #that when a checkpoint is created this value is saved.
  #Making the plots in tensorboard being continued when the model is restored.
  global_step = tf.Variable(0)
  increment_step = global_step.assign_add(1)

  #Create a queue that will be automatically fill by another thread
  #as we read batches out of it
  batch = batch_queue(filename_list, batch_size)

  # Create the graph, etc.
  m = Model(batch)

  init_op = tf.initialize_all_variables()
  #This is required to intialize num_epochs for the filename_queue
  init_local = tf.initialize_local_variables()

  # Create a session for running operations in the Graph.
  sess = tf.Session(config=tf.ConfigProto(
    log_device_placement=False, allow_soft_placement=True))

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
          summary =  m.summarize(writer).eval(session=sess)
          writer.add_summary(summary, global_step=global_step)
          
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

def batch_queue(filenames, batch_size, num_epochs=None):

  with tf.variable_scope("batch_queue"):
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

def pyramid(tensor, stop_shape, name="pyramid"):
  with tf.variable_scope('pyramid'):
    _pyramid = [tensor]
    while _pyramid[-1]._shape[1] > stop_shape:
      new_tensor = tf.nn.avg_pool(_pyramid[-1], [1,2,2,1],[1,2,2,1], 'VALID', name='pyramid_{}'.format(new_tensor._shape[1]))
      _pyramid.append(new_tensor)
    return _pyramid

def pyramid_random_crop(tensor, stop_shape):
  with tf.variable_scope('pyramid_random_crop'):
    crop_shape = map(int, tensor._shape)
    crop_shape[1:3] = [stop_shape,stop_shape]
    _crops = [tf.random_crop(tensor, crop_shape, seed=None, name=None)] 
    while tensor._shape[1] > stop_shape:
      tensor = tf.nn.avg_pool(tensor, [1,2,2,1],[1,2,2,1], 'VALID', name='pyramid_{}'.format(tensor._shape[1]/2))
      _crops.append(tf.random_crop(tensor, crop_shape, seed=None, name=None))
    return _crops

class Model(object):

  def __init__(self, batch, z_size=100, learning_rate=0.0002, beta1=0.5):
    """
    Learning rate of for adam
    Momentum term of adam
    """
    self.batch_size, self.c_dim = int(batch._shape[0]), int(batch._shape[3]) 
    real_images_pyramid = pyramid_random_crop(batch, 32)

    with tf.variable_scope("model"):
      self.z = tf.random_uniform(shape=(self.batch_size, z_size), 
                            minval=-1.0, maxval=1.0, 
                            dtype=tf.float32, seed=None, name='z')

      fake_images = self.generator(self.z)
      fake_images_pyramid = pyramid_random_crop(fake_images, 32)

      d_real, d_logits_real = self.discriminator(real_images_pyramid[-1], name='discriminator_32')
      d_fake, d_logits_fake = self.discriminator(fake_images_pyramid[-1], name='discriminator_32',reuse=True)


      with tf.variable_scope('losses'):

        #discriminator losses
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(d_logits_real,
                                                    tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake,
                                                    tf.zeros_like(d_fake)))
        d_loss_total = d_loss_real + d_loss_fake

        #generator loss
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake,
                                                    tf.ones_like(d_fake)))

      with tf.variable_scope('trainning'):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                          .minimize(d_loss_total, var_list=d_vars)
        
        g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                          .minimize(g_loss, var_list=g_vars)

      with tf.variable_scope("summary"):

        scalars = [tf.scalar_summary(name, var) for name, var in [("d_loss_real",d_loss_real),
                                                                  ("d_loss_fake",d_loss_fake),
                                                                  ("d_loss_total",d_loss_total),
                                                                  ("g_loss_total",g_loss)]]
        hist = [tf.histogram_summary(name, var) for name, var in [("d_real", d_real),("d_fake", d_fake)]]
        crop_real = [tf.image_summary('crop_real_{:03d}'.format(2**(8-i)),image,max_images=1) for i,image in  enumerate(real_images_pyramid)]
        crop_fake = [tf.image_summary('crop_fake_{:03d}'.format(2**(8-i)),image,max_images=1) for i,image in  enumerate(fake_images_pyramid)]
        self.summary = tf.merge_summary(scalars+crop_real+crop_fake+hist)


  def step(self, sess):
    sess.run([self.d_optim, self.g_optim])

  def summarize(self, writer):
    return self.summary


  def generator(self, z, gf_dim=64):
    with tf.variable_scope("generator"):
      z_ = linear(z, gf_dim*4*4*4*8, 'g_h0_lin')

      with tf.variable_scope("layer_0"):
        h0 = tf.reshape(z_, [self.batch_size, 4, 4, gf_dim * 32])
        g_bn0 = batch_norm(name='g_bn0')
        h0 = tf.nn.relu(g_bn0(h0))

      with tf.variable_scope("layer_1"):
        h1 = conv2d_transpose(h0,
            [self.batch_size, 8, 8, gf_dim*16], name='g_h1')
        g_bn1 = batch_norm(name='g_bn1')
        h1 = tf.nn.relu(g_bn1(h1))

      with tf.variable_scope("layer_2"):
        h2 = conv2d_transpose(h1,
            [self.batch_size, 16, 16, gf_dim*8], name='g_h2')
        g_bn2 = batch_norm(name='g_bn2')
        h2 = tf.nn.relu(g_bn2(h2))

      with tf.variable_scope("layer_3"):
        h3 = conv2d_transpose(h2,
            [self.batch_size, 32, 32, gf_dim*4], name='g_h3')
        g_bn3 = batch_norm(name='g_bn3')
        h3 = tf.nn.relu(g_bn3(h3))        
      
      with tf.variable_scope("layer_4"):
        h4 = conv2d_transpose(h3,
            [self.batch_size, 64, 64, gf_dim*2], name='g_h4')
        g_bn4 = batch_norm(name='g_bn4')
        h4 = tf.nn.relu(g_bn4(h4))

      with tf.variable_scope("layer_5"):
        h5 = conv2d_transpose(h4,
            [self.batch_size, 128, 128, gf_dim*1], name='g_h5')
        g_bn5 = batch_norm(name='g_bn5')
        h5 = tf.nn.relu(g_bn5(h5))
      
      with tf.variable_scope("layer_6"):
        h6 = conv2d_transpose(h5,
            [self.batch_size, 256, 256, self.c_dim], name='g_h6')
        return tf.nn.tanh(h6)

  def discriminator(self, image, df_dim=64, reuse=False, name="discriminator"):
    
    with tf.variable_scope(name, reuse=reuse):
      with tf.variable_scope("layer_0"):
        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        d_bn1 = batch_norm(name='d_bn1')
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))

      with tf.variable_scope("layer_1"):
        d_bn2 = batch_norm(name='d_bn2')
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
      
      with tf.variable_scope("layer_2"):
        h3 = linear(tf.reshape(h2, [-1, 4096]), 1, name='d_h3_lin')
        return tf.nn.sigmoid(h3), h3

if __name__ == '__main__':
  filename_list = list(glob('/usr/people/it2/seungmount/research/datasets/lsum_png_copy/*.png'))
  with tf.device("/gpu:1"):
    run(filename_list,64, Model)