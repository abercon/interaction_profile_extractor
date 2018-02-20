import tensorflow as tf

from os import getcwd
from data_ops import DataStore
from prefrences import Prefrences
from ops import conv2d, fully_connected, conv_transpose

class Vae():

    def __init__(self):
        self.store = DataStore()
        self.pref = Prefrences()
        self.heatmaps = tf.placeholder(shape = [None, self.pref.windowsize, self.pref.windowsize,1], dtype=tf.float32)

        z_mean, z_stddev = self._encode(self.heatmaps)
        gaussian_samples = tf.random_normal([self.pref.batch_size, self.pref.n_z], self.pref.sampling_mean, self.pref.sampling_stddev)
        noisy_z = z_mean + (z_stddev*gaussian_samples)
        
        generated_heatmaps = self._decode(noisy_z)
        flattened_gen_hm = tf.reshape(generated_heatmaps, [self.pref.batch_size, self.pref.windowsize**2])
        flattened_heatmaps = tf.reshape(self.heatmaps, [self.pref.batch_size, self.pref.windowsize**2])
        
        self.generative_loss = -tf.reduce_sum(flattened_heatmaps * tf.log(1e-8 + flattened_gen_hm) + (1-flattened_heatmaps) * tf.log(1e-8 + 1 - flattened_gen_hm),1)
        self.kl_divergence = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        
        self.loss = tf.reduce_mean(self.generative_loss + self.kl_divergence)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def _encode(self, training_data):
        with tf.variable_scope("encoder"):
            f1 = tf.nn.relu(conv2d("1st_conv", training_data, 1, 50, self.pref))
            f2 = tf.nn.relu(conv2d("2nd_conv", f1, 50, 100, self.pref))
            f3 = tf.nn.relu(conv2d("3rd_conv", f2, 100, 200, self.pref))
    
            flat_f3 = tf.reshape(f3, [self.pref.batch_size, 13*13*200])
    
            z_means = fully_connected("enc_means", flat_f3, 13*13*200, self.pref.n_z)
            z_stddev = fully_connected("enc_stddev", flat_f3, 13*13*200, self.pref.n_z)

        return z_means, z_stddev

    def _decode(self, latent_variables):
        with tf.variable_scope("decoder"):   
            z_expanded = fully_connected("dec_expansion", latent_variables, self.pref.n_z, 13*13*200)
            z_shaped = tf.reshape(z_expanded, [self.pref.batch_size, 13, 13, 200]) 
            
            dec_f3 = tf.nn.relu(conv_transpose("1st_deconv", z_shaped, [self.pref.rfs, self.pref.rfs, self.pref.n_z, z_shaped.get_shape()[-1]], [self.pref.batch_size, 25, 25, 100], self.pref))
            dec_f2 = tf.nn.relu(conv_transpose("2nd_deconv", dec_f3, [self.pref.rfs, self.pref.rfs, 50, dec_f3.get_shape()[-1]], [self.pref.batch_size, 50, 50, 50], self.pref))
            dec_f1 = tf.nn.sigmoid(conv_transpose("3rd_deconv", dec_f2, [self.pref.rfs, self.pref.rfs, 1, dec_f2.get_shape()[-1]], [self.pref.batch_size, 100, 100, 1], self.pref))
        
        return dec_f1
    
    def train(self):
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for batches in range(len(self.store.train.data)):
                batch = self.store.train.get_next_batch()
                optimiser, generation_loss, latent_loss = sess.run((self.optimizer, self.generative_loss,  self.kl_divergence), feed_dict={self.heatmaps:batch})  
                saver.save(sess, getcwd()+"/training/saves", global_step=0)
                print (self.store.train.batch_number, ':', self.loss)




vae = Vae()
vae.train()
                
            