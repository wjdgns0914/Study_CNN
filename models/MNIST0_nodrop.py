from nnUtils import *

model = Sequential([
    SpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution'),
    ReLU(name='L2_ReLU'),
    SpatialMaxPooling(2,2,2,2,name='L3_MaxPooling',padding='SAME'),

    SpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L4_Convolution'),
    ReLU(name='L5_ReLU'),
    SpatialMaxPooling(2,2,2,2,name='L6_MaxPooling',padding='SAME'),

    SpatialConvolution(128, 3, 3, padding='SAME', bias=False, name='L7_Convolution'),
    ReLU(name='L8_ReLU'),
    SpatialMaxPooling(2,2,2,2,name='L9_MaxPooling',padding='SAME'),

    Affine(625, bias=False,name='L10_FullyConnected'),
    ReLU(name='L11_ReLU'),

    Affine(10,bias=False,name='L12_FullyConnected'),
])
"""
W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.get_variable("W3", shape=[3, 3,64,128],initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.relu(tf.matmul(L3_flat, W4))

W5 = tf.get_variable("W5", shape=[625, 10],initializer=tf.contrib.layers.xavier_initializer())
logits = tf.matmul(L4, W5)

"""