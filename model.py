# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange

from ops import *
from utils import *
import cPickle as pickle

class DCGAN(object):
    def __init__(self, sess, image_size=256, is_crop=False,
                 batch_size=128, sample_size=128,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1, piramid=(32,) ):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
            lam: relative importance of contextual and perceptual loss
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size #Shouldn't batch_size and sample_size be the same?
        self.c_dim = c_dim
        self.image_shape = [image_size, image_size, c_dim]
        self.piramid = piramid
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim
        self.gfc_dim = gfc_dim
        self.checkpoint_dir = checkpoint_dir
        self.lam = lam




        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')

        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.histogram_summary("z", self.z)

        self.images_32 = tf.nn.avg_pool(self.images, [1,8,8,1],[1,8,8,1], 'VALID',  name='real_images_32')
        self.sample_images_32 = tf.nn.avg_pool(self.sample_images, [1,8,8,1],[1,8,8,1], 'VALID',  name='sample_images_32')

        self.G = self.generator(self.z)
        self.G_32 = tf.nn.avg_pool(self.G, [1,8,8,1],[1,8,8,1], 'VALID')
        self.D, self.D_logits = self.discriminator(self.images_32)

        self.sampler = self.sampler(self.z)
        self.sampler_32 = tf.nn.avg_pool(self.sampler, [1,8,8,1],[1,8,8,1], 'VALID')
        self.D_, self.D_logits_ = self.discriminator(self.G_32, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_32_sum = tf.image_summary("G_32", self.G_32)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                                    tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        self.saver = tf.train.Saver(max_to_keep=10)

        # Completion.
        self.mask = tf.placeholder(tf.float32, [None] + [self.piramid[0],self.piramid[0], self.c_dim], name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.mul(self.mask, self.G_32) - tf.mul(self.mask, self.images_32))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)


    def _build_sampling_constants(self,data):
        """
        When sampling, we will always use the same seeds, and the same sample files to compute the loss
        """
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_z, sample_images

    def _build_batch(self, data, idx):
        """
        Pretty much the same as _build_sampling_constants
        """
        batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
        batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                 for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                    .astype(np.float32)
        return batch_z, batch_images           

    def train(self, config):
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary(
            [self.z_sum, self.d__sum, self.G_sum, self.G_32_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        data = glob(os.path.join(config.dataset, "*.png"))
        #np.random.shuffle(data)
        assert(len(data) > 0)
        batch_idxs = min(len(data), config.train_size) // self.batch_size
        sample_z, sample_images = self._build_sampling_constants(data)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            for idx in xrange(batch_idxs):
          

                batch_z, batch_images = self._build_batch(data, idx)

                # Update D network
                _, summary_str_d, _ , summary_str_g = self.sess.run([d_optim, self.d_sum, g_optim, self.g_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str_d, counter)
                self.writer.add_summary(summary_str_g, counter)

                        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str, errD_fake, errD_real, errG = self.sess.run(
                    [g_optim, self.g_sum, self.d_loss_fake, self.d_loss_real, self.g_loss],
                    feed_dict={ self.z: batch_z, self.images: batch_images })
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))
summary_str_g
                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler_32, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    save_images(samples, [8, self.sample_size/8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


    def find_best_z(self, config,  zhats, batch_mask, batch_images, batchSz, v, masked_images, nIter=100):

        for i in xrange(nIter):
            fd = {
                self.z: zhats,
                self.mask: batch_mask,
                self.images: batch_images,
            }
            run = [self.contextual_loss , self.perceptual_loss, self.complete_loss, self.grad_complete_loss, self.G_32]
            contextual, perceptual, loss, g, G_imgs = self.sess.run(run, feed_dict=fd)
            assert g[0].shape[0] == batchSz, "I would call this a bug"

            v_prev = np.copy(v)
            v = config.momentum*v - config.lr*g[0]
            zhats += -config.momentum * v_prev + (1+config.momentum)*v
            np.clip(zhats, -1, 1)

            if i % 50 == 0:
                print('iteratation: {}, complete loss {}, contextual {},  perceptual {}'.format(
                    i, np.mean(loss[0:batchSz]),
                       np.mean(contextual[0:batchSz]),
                       perceptual))
                # imgName = os.path.join(config.outDir,
                #                        'hats_imgs/{:04d}.png'.format(i))
                # nRows = np.ceil(batchSz/8)
                # nCols = 8
                # save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                # inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                # completeed = masked_images + inv_masked_hat_images
                # imgName = os.path.join(config.outDir,
                #                        'completed/{:04d}.png'.format(i))
                # save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)

                # #save gradients
                # images = contextual - np.min(contextual)
                # images = images / np.max(images)
                # images = images[:,np.newaxis,np.newaxis] * np.ones(shape=(128,32,32))
                # images = images[:,:,:,np.newaxis]
                # imgName = os.path.join(config.outDir,
                #                        'gradients/{:04d}.png'.format(i))
                # save_images(images[:batchSz,:,:,:], [nRows,nCols], imgName)

        return zhats, loss


    def _create_mask(self, config):

        if config.maskType == 'random':
            assert(False)
        elif config.maskType == 'center':
            scale = 0.35
            assert(scale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*scale)
            u = int(self.image_size*(1.0-scale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            assert(False)
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        else:
            assert(False)

        return mask

    def complete(self, config):
        np.set_printoptions(suppress=True)

        if not os.path.exists(os.path.join(config.outDir, 'hats_imgs')):
            os.makedirs(os.path.join(config.outDir, 'hats_imgs'))
        if not os.path.exists(os.path.join(config.outDir, 'completed')):
            os.makedirs(os.path.join(config.outDir, 'completed'))
        if not os.path.exists(os.path.join(config.outDir, 'gradients')):
            os.makedirs(os.path.join(config.outDir, 'gradients'))
        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
   
        mask = self._create_mask(config)


        for idx in xrange(0, batch_idxs):
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)

            nRows = np.ceil(batchSz/8)
            nCols = 8
            # save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
            #             os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, batch_mask)
            # save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
            #             os.path.join(config.outDir, 'masked.png'))

            n_sampling = 10
            all_contextual = np.empty(shape=(self.batch_size, n_sampling))
            all_z = np.empty(shape=(self.batch_size, self.z_dim, n_sampling))
            for sample in xrange(n_sampling):
                zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
                v = 0
                all_z[:,:,sample] , all_contextual[:,sample] = self.find_best_z(config, zhats, batch_mask, batch_images, batchSz, v, masked_images)

            best_sample = np.argmin(all_contextual, axis=1)
            best_z = np.empty(shape=(self.batch_size, self.z_dim))
            best_loss = np.empty(shape=(self.batch_size, 1))
            for i in range(self.batch_size):
                best_z[i,:] = all_z[i,:,best_sample[i]]
                best_loss =  all_contextual[i, best_sample[i]]

            print np.mean(best_loss)
            v = 0
            zhats = self.find_best_z(config, best_z , batch_mask, batch_images, batchSz, v, masked_images, nIter=config.nIter)
            with open('./completions/{}.p'.format(idx),'wb') as f:
                pickle.dump(zhats,f)

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = linear(tf.reshape(h2, [-1, 4096]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3

    def generator(self, z):
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*4*4*4*8, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [self.batch_size, 4, 4, self.gf_dim * 32])
        h0 = tf.nn.relu(self.g_bn0(self.h0))
        print h0._shape

        self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
            [self.batch_size, 8, 8, self.gf_dim*16], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
            [self.batch_size, 16, 16, self.gf_dim*8], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
            [self.batch_size, 32, 32, self.gf_dim*4], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))        

        h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
            [self.batch_size, 64, 64, self.gf_dim*2], name='g_h4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(h4))

        h5, self.h5_w, self.h5_b = conv2d_transpose(h4,
            [self.batch_size, 128, 128, self.gf_dim*1], name='g_h5', with_w=True)
        h5 = tf.nn.relu(self.g_bn5(h5))

        h6, self.h6_w, self.h6_b = conv2d_transpose(h5,
            [self.batch_size, 256, 256, self.c_dim], name='g_h6', with_w=True)
        
        print h6._shape
        return tf.nn.tanh(h6)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        h0 = tf.reshape(linear(z, self.gf_dim*4*4*4*8, 'g_h0_lin'),[self.batch_size, 4, 4, self.gf_dim * 32])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim*16], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
            [self.batch_size, 16, 16, self.gf_dim*8], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
            [self.batch_size, 32, 32, self.gf_dim*4], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))        

        h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
            [self.batch_size, 64, 64, self.gf_dim*2], name='g_h4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(h4))

        h5, self.h5_w, self.h5_b = conv2d_transpose(h4,
            [self.batch_size, 128, 128, self.gf_dim*1], name='g_h5', with_w=True)
        h5 = tf.nn.relu(self.g_bn5(h5))

        h6, self.h6_w, self.h6_b = conv2d_transpose(h5,
            [self.batch_size, 256, 256, self.c_dim], name='g_h6', with_w=True)
 
        return tf.nn.tanh(h6)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
