import tensorflow as tf

def conv2d(name, x, features_in, features_out, pref):
    with tf.variable_scope(name):
        conv_biases = tf.get_variable("biases", shape=[features_out], initializer=tf.constant_initializer(0.0))
        conv_weights = tf.get_variable("weights", shape = [pref.rfs, pref.rfs, features_in, features_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        features = tf.nn.conv2d(x, conv_weights, strides = [1,pref.conv_stride,pref.conv_stride,1], padding="SAME") + conv_biases
        print (name + ' output shape:', features.shape)
    return features

def fully_connected(name, values, n_features_in, n_z_out,):
    with tf.variable_scope(name):
        fc_weights = tf.get_variable('weights', [n_features_in, n_z_out], initializer = tf.random_normal_initializer(stddev=0.2))
        fc_biases = tf.get_variable('biases', [n_z_out], initializer = tf.constant_initializer())
    return tf.matmul(values, fc_weights) + fc_biases

def conv_transpose(name, values, input_shape, output_shape, pref):
    with tf.variable_scope(name):
        transpose_weights = tf.get_variable('weights', shape = input_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        transpose_biases = tf.get_variable('biases', shape = [output_shape[-1]], initializer = tf.constant_initializer(0.0))
        transposition = tf.nn.conv2d_transpose(values, transpose_weights, output_shape = output_shape, strides = [1, pref.conv_stride, pref.conv_stride, 1])
    return tf.reshape(tf.nn.bias_add(transposition, transpose_biases), transposition.get_shape())

#shape = [pref.rfs, pref.rfs, pref.n_z, values.get_shape()[-1]]