from nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2
model = Sequential([
    SpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution'),
    BatchNormalization(name='L2_Batch'),
    HardTanh(name='L3_HardTanh'),

    SpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L4_Convolution'),
    BatchNormalization(name='L5_Batch'),
    HardTanh(name='L6_HardTanh'),
    SpatialMaxPooling(2, 2, 2, 2, name='L7_MaxPooling', padding='SAME'),

    SpatialConvolution(128, 3, 3, padding='SAME', bias=False, name='L8_Convolution'),
    BatchNormalization(name='L9_Batch'),
    HardTanh(name='L10_HardTanh'),
    SpatialMaxPooling(2, 2, 2, 2, name='L11_MaxPooling', padding='SAME'),

    SpatialConvolution(256, 3, 3, padding='SAME', bias=False, name='L12_Convolution'),
    BatchNormalization(name='L13_Batch'),
    HardTanh(name='L14_HardTanh'),
    SpatialMaxPooling(2,2,2,2,name='L15_MaxPooling',padding='SAME'),

    Affine(625, bias=False,name='L16_FullyConnected'),
    BatchNormalization(name='L17_Batch'),
    HardTanh(name='L18_HardTanh'),

    Affine(10,bias=False,name='L19_FullyConnected'),
])