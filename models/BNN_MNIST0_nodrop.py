from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(32,3,3, padding='SAME', bias=False,name='L1_Convolution'),
    ReLU(name='L2_ReLU'),
    SpatialMaxPooling(2,2,2,2,name='L3_MaxPooling',padding='SAME'),

    BinarizedSpatialConvolution(64, 3, 3, padding='SAME', bias=False, name='L4_Convolution'),
    ReLU(name='L5_ReLU'),
    SpatialMaxPooling(2,2,2,2,name='L6_MaxPooling',padding='SAME'),

    BinarizedSpatialConvolution(128, 3, 3, padding='SAME', bias=False, name='L7_Convolution'),
    ReLU(name='L8_ReLU'),
    SpatialMaxPooling(2,2,2,2,name='L9_MaxPooling',padding='SAME'),

    BinarizedAffine(625, bias=False,name='L10_FullyConnected'),
    ReLU(name='L11_ReLU'),

    BinarizedAffine(10,bias=False,name='L12_FullyConnected'),
])
