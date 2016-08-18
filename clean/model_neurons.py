import tensorflow as tf
from model_mnist import Model

class ModelNeurons(Model):
  def create_summaries(self, d_loss_real, d_loss_fake, g_loss, d_real, d_fake, real_images, fake_images):
    with tf.variable_scope("summary"):

      scalars = [tf.scalar_summary(name, var) for name, var in [("d_loss_real",d_loss_real),
                                                                  ("d_loss_fake",d_loss_fake),
                                                                  ("d_loss_total",self.d_loss_total),
                                                                  ("g_loss_total",g_loss)]]
      hist = [tf.histogram_summary(name, var) for name, var in [("d_real", d_real),("d_fake", d_fake),("z",self.z)]]
      fullres = [tf.image_summary(name, tf.reshape(var,shape=(self.batch_size, 32, 32*32, 1))) for name, var in [("fullres_real", real_images),("fullres_fake", fake_images)]]
      self.summary = tf.merge_summary(scalars+fullres+hist)