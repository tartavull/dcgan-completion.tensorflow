from ops import *

class Model(object):

  def __init__(self, real_images, z_size=100, learning_rate=0.0002, beta1=0.5, train=True):
    """
    Learning rate of for adam
    Momentum term of adam
    Train should be false if we are doing inference
    """
    self.train =  train
    self.batch_size, self.c_dim = int(real_images._shape[0]), int(real_images._shape[3]) 

    with tf.variable_scope("model"):
      self.z = tf.random_uniform(shape=(self.batch_size, z_size), 
                            minval=-1.0, maxval=1.0, 
                            dtype=tf.float32, seed=None, name='z')

      fake_images = self.generator(self.z)

      d_real, d_logits_real = self.discriminator(real_images, name='discriminator')
      d_fake, d_logits_fake = self.discriminator(fake_images, name='discriminator',reuse=True)


      with tf.variable_scope('losses'):

        #discriminator losses
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(d_logits_real,
                                                    tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake,
                                                    tf.zeros_like(d_fake)))
        self.d_loss_total = d_loss_real + d_loss_fake

        #generator loss
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake,
                                                    tf.ones_like(d_fake)))

      with tf.variable_scope('trainning'):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                          .minimize(self.d_loss_total, var_list=d_vars)
        
        g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                          .minimize(g_loss, var_list=g_vars)

      self.create_summaries(d_loss_real, d_loss_fake, g_loss, d_real, d_fake, real_images, fake_images)
     


  def create_summaries(self, d_loss_real, d_loss_fake, g_loss, d_real, d_fake, real_images, fake_images):
    with tf.variable_scope("summary"):

      scalars = [tf.scalar_summary(name, var) for name, var in [("d_loss_real",d_loss_real),
                                                                  ("d_loss_fake",d_loss_fake),
                                                                  ("d_loss_total",self.d_loss_total),
                                                                  ("g_loss_total",g_loss)]]
      hist = [tf.histogram_summary(name, var) for name, var in [("d_real", d_real),("d_fake", d_fake),("z",self.z)]]
      fullres = [tf.image_summary(name, var) for name, var in [("fullres_real", real_images),("fullres_fake", fake_images)]]
      self.summary = tf.merge_summary(scalars+fullres+hist)

  def step(self, sess):
    d_loss, _ , _ , _ = sess.run([self.d_loss_total, self.d_optim, self.g_optim, self.z])
    while d_loss < 0.6:
      d_loss, _ = sess.run([self.d_loss_total, self.g_optim])

  def summarize(self, sess, writer, global_step):
    summary = sess.run(self.summary)
    writer.add_summary(summary, global_step=global_step)


  def generator(self, z, gf_dim=64):
    with tf.variable_scope("generator"):
      z_ = linear(z, gf_dim*4*4*4, 'g_h0_lin')

      with tf.variable_scope("layer_0"):
        h0 = tf.reshape(z_, [self.batch_size, 4, 4, gf_dim * 4])
        g_bn0 = batch_norm(name='g_bn0')
        h0 = tf.nn.relu(g_bn0(h0, train=self.train))

      with tf.variable_scope("layer_1"):
        h1 = conv2d_transpose(h0,
            [self.batch_size, 8, 8, gf_dim*2], name='g_h1')
        g_bn1 = batch_norm(name='g_bn1')
        h1 = tf.nn.relu(g_bn1(h1, train=self.train))

      with tf.variable_scope("layer_2"):
        h2 = conv2d_transpose(h1,
            [self.batch_size, 16, 16, gf_dim*1], name='g_h2')
        g_bn2 = batch_norm(name='g_bn2')
        h2 = tf.nn.relu(g_bn2(h2, train=self.train))

      with tf.variable_scope("layer_3"):
        h3 = conv2d_transpose(h2,
            [self.batch_size, 32, 32, self.c_dim], name='g_h3')
        return tf.nn.tanh(h3)      

  def discriminator(self, image, df_dim=64, reuse=False, name="discriminator"):
    
    with tf.variable_scope(name, reuse=reuse):
      with tf.variable_scope("layer_0"):
        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        d_bn1 = batch_norm(name='d_bn1')
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv'), train=self.train))

      with tf.variable_scope("layer_1"):
        d_bn2 = batch_norm(name='d_bn2')
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv'), train=self.train))
      
      with tf.variable_scope("layer_2"):
        h3 = linear(tf.reshape(h2, [-1, 4096]), 1, name='d_h3_lin')
        return tf.nn.sigmoid(h3), h3
