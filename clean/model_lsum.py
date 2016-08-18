from ops import *

class Model(object):

  def __init__(self, batch, z_size=100, learning_rate=0.0002, beta1=0.5, train=True):
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

      losses = {}
      for i in range(len(real_images_pyramid)):
        level = str(2**(8-i))
        d_real, d_logits_real = self.discriminator(real_images_pyramid[i], name='discriminator_'+level)
        d_fake, d_logits_fake = self.discriminator(fake_images_pyramid[i], name='discriminator_'+level,reuse=True)
        
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
          g_loss_total = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake,
                                                       tf.ones_like(d_fake)))

          losses['d_loss_real_'+level] = d_loss_real
          losses['d_logits_fake_'+level] = d_loss_fake
          losses['d_loss_total_'+level] = d_loss_total
          losses['g_loss_total_'+level]  = g_loss_total

      with tf.variable_scope('training'):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        
        self.d_loss_total = 0.0 * losses['d_loss_total_32'] + 0.0 * losses['d_loss_total_64'] \
                            + 0.0 * losses['d_loss_total_128'] +  1.0 * losses['d_loss_total_256']
        self.g_loss_total = 0.0 * losses['g_loss_total_32'] + 0.0 * losses['g_loss_total_64'] \
                            + 0.0 * losses['g_loss_total_128'] + 1.0 * losses['g_loss_total_256']

        losses['d_loss_total_weighted_sum'] = self.d_loss_total   
        losses['g_loss_total_weighted_sum'] = self.g_loss_total                                    
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                          .minimize(self.d_loss_total, var_list=d_vars)
        
        g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                          .minimize(self.g_loss_total, var_list=g_vars)

      with tf.variable_scope("summary"):

        scalars = [tf.scalar_summary(name, var) for name, var in losses.iteritems()]
        hist = [tf.histogram_summary(name, var) for name, var in [("d_real", d_real),("d_fake", d_fake),("z",self.z)]]
        crop_real = [tf.image_summary('crop_real_{:03d}'.format(2**(8-i)),
          image,max_images=1) for i,image in  enumerate(real_images_pyramid)]
        crop_fake = [tf.image_summary('crop_fake_{:03d}'.format(2**(8-i)),
          image,max_images=1) for i,image in  enumerate(fake_images_pyramid)]
        fullres = [tf.image_summary(name, var) for name, var in [("fullres_real", batch),("fullres_fake", fake_images)]]
        self.summary = tf.merge_summary(scalars+crop_real+crop_fake+fullres+hist)


  def step(self, sess):
    d_loss, _ , _ , _ = sess.run([self.d_loss_total, self.d_optim, self.g_optim, self.z])
    while d_loss < 0.6:
      d_loss, _ = sess.run([self.d_loss_total, self.g_optim])

  def summarize(self, sess, writer, global_step):
    summary = sess.run(self.summary)
    writer.add_summary(summary, global_step=global_step)


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
