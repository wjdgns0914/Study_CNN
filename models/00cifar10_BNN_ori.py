#00: 1차,2차 이때 쯤에  쓰던 모델, 아마 논문에서 따왔던걸로 기억한다.
from nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2
model = Sequential([
    BinarizedWeightOnlySpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution',Drift=Dri1),
    BatchNormalization(name='L2_Batch'),
    # ReLU(name='L3_ReLU'),
    HardTanh(name='L3_HardTanh'),
    BinarizedSpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L4_Convolution',Drift=Dri1),
    SpatialMaxPooling(2,2,2,2,name='L5_MaxPooling',padding='SAME'),

    # ReLU(name='L5_ReLU'),

    BatchNormalization(name='L6_Batch'),
    HardTanh(name='L7_HardTanh'),
    # ReLU(name='L7_ReLU'),
    BinarizedSpatialConvolution(128, 3, 3, padding='SAME', bias=False, name='L8_Convolution',Drift=Dri1),
    SpatialMaxPooling(2,2,2,2,name='L9_MaxPooling',padding='SAME'),
    # ReLU(name='L8_ReLU'),

    BatchNormalization(name='L10_Batch'),
    HardTanh(name='L11_HardTanh'),
    # ReLU(name='L11_ReLU'),
    BinarizedSpatialConvolution(256, 3, 3, padding='SAME', bias=False, name='L12_Convolution',Drift=Dri1),
    SpatialMaxPooling(2,2,2,2,name='L13_MaxPooling',padding='SAME'),
    # ReLU(name='L11_ReLU'),

    BatchNormalization(name='L14_Batch'),
    HardTanh(name='L15_HardTanh'),
    # ReLU(name='L15_ReLU'),
    BinarizedAffine(625, bias=False,name='L16_FullyConnected',Drift=Dri2),
    # ReLU(name='L14_ReLU'),

    BatchNormalization(name='L17_Batch'),
    HardTanh(name='L18_HardTanh'),
    # ReLU(name='L18_ReLU'),
    BinarizedAffine(10,bias=False,name='L19_FullyConnected',Drift=Dri2),
])