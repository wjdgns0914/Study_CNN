# 05: 04보다 네트워크가 한층 더 커진다. 필터수도 훨씬 많고
# 풀링레이어도 2개에서 3개로 늘어났다. 그리고 FC layer는 한개지만 노드 수는 1024개로 늘렸다.
# 노드 수는 많아졌지만 풀링을 세번하기 때문에 파라미터 수가 그렇게 많아지지는 않는다.
from nnUtils import *
FLAGS = tf.app.flags.FLAGS
print("Model")
print(FLAGS.Drift1,FLAGS.Drift2,FLAGS.Variation)
# Dri1=FLAGS.Drift1
# Dri2=FLAGS.Drift2
Dri1=False
Dri2=False
model = Sequential([
    BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', bias=False,name='L1_Convolution',Drift=Dri1),
    BatchNormalization(name='L2_Batch'),
    ReLU(name='L3_ReLU'),

    BinarizedWeightOnlySpatialConvolution(128,3,3, padding='SAME', bias=False,name='L4_Convolution',Drift=Dri1),
    BatchNormalization(name='L5_Batch'),
    ReLU(name='L6_ReLU'),

    SpatialMaxPooling(2, 2, 2, 2, name='L7_MaxPooling', padding='SAME'),

    BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', bias=False,name='L8_Convolution',Drift=Dri1),
    BatchNormalization(name='L9_Batch'),
    ReLU(name='L10_ReLU'),

    BinarizedWeightOnlySpatialConvolution(256,3,3, padding='SAME', bias=False,name='L11_Convolution',Drift=Dri1),
    BatchNormalization(name='L12_Batch'),
    ReLU(name='L13_ReLU'),

    SpatialMaxPooling(2, 2, 2, 2, name='L14_MaxPooling', padding='SAME'),

    BinarizedWeightOnlySpatialConvolution(512, 3, 3, padding='SAME', bias=False, name='L15_Convolution', Drift=Dri1),
    BatchNormalization(name='L16_Batch'),
    ReLU(name='L17_ReLU'),

    BinarizedWeightOnlySpatialConvolution(512, 3, 3, padding='SAME', bias=False,name='L18_Convolution',Drift=Dri1),
    BatchNormalization(name='L19_Batch'),
    ReLU(name='L20_ReLU'),

    SpatialMaxPooling(2,2,2,2,name='L21_MaxPooling',padding='SAME'),

    BinarizedWeightOnlyAffine(1024, bias=False,name='L22_FullyConnected',Drift=Dri2),
    BatchNormalization(name='L23_Batch'),
    ReLU(name='L24_HardTanh'),

    BinarizedWeightOnlyAffine(10,bias=False,name='L25_FullyConnected',Drift=Dri2),
])