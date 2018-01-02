from nnUtils import *
#04: 03의 바이너리 연산을 하는 네트워크 02(01의 바이너리버전)보다 컨볼루션 필터 크기가 약간 더 크다.
#FC layer를 한개로 줄였다.
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
# Dri1=FLAGS.Drift1
# Dri2=FLAGS.Drift2
Dri1=False
Dri2=False
model = Sequential([
    BinarizedWeightOnlySpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution',Drift=Dri1),
    BatchNormalization(name='L2_Batch'),
    HardTanh(name='L3_HardTanh'),

    BinarizedWeightOnlySpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L4_Convolution',Drift=Dri1),
    BatchNormalization(name='L5_Batch'),
    HardTanh(name='L6_HardTanh'),
    SpatialMaxPooling(2, 2, 2, 2, name='L7_MaxPooling', padding='SAME'),

    BinarizedWeightOnlySpatialConvolution(128, 3, 3, padding='SAME', bias=False, name='L8_Convolution',Drift=Dri1),
    BatchNormalization(name='L9_Batch'),
    HardTanh(name='L10_HardTanh'),
    SpatialMaxPooling(2, 2, 2, 2, name='L11_MaxPooling', padding='SAME'),

    BinarizedWeightOnlySpatialConvolution(256, 3, 3, padding='SAME', bias=False, name='L12_Convolution',Drift=Dri1),
    BatchNormalization(name='L13_Batch'),
    HardTanh(name='L14_HardTanh'),
    SpatialMaxPooling(2,2,2,2,name='L15_MaxPooling',padding='SAME'),

    BinarizedWeightOnlyAffine(625, bias=False,name='L16_FullyConnected',Drift=Dri2),
    BatchNormalization(name='L17_Batch'),
    HardTanh(name='L18_HardTanh'),

    BinarizedWeightOnlyAffine(10,bias=False,name='L19_FullyConnected',Drift=Dri2),
])