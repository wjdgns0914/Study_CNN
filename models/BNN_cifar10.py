from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(128,3,3,1,1, padding='VALID', bias=False,name='L1_Convolution'),
    BatchNormalization(name='L2_Batch'),
    HardTanh(name='L3_HardTanh'),

    BinarizedSpatialConvolution(128,3,3, padding='SAME', bias=False,name='L4_Convolution'),
    SpatialMaxPooling(2,2,2,2,name='L5_MaxPooling'),
    BatchNormalization(name='L6_Batch'),
    HardTanh(name='L7_HardTanh'),

    BinarizedSpatialConvolution(256,3,3, padding='SAME', bias=False,name='L8_Convolution'),
    BatchNormalization(name='L9_Batch'),
    HardTanh(name='L10_HardTanh'),

    BinarizedSpatialConvolution(256,3,3, padding='SAME', bias=False,name='L11_Convolution'),
    SpatialMaxPooling(2,2,2,2,name='L12_MaxPooling'),
    BatchNormalization(name='L13_Batch'),
    HardTanh(name='L14_HardTanh'),

    BinarizedSpatialConvolution(512,3,3, padding='SAME', bias=False,name='L15_Convolution'),
    BatchNormalization(name='L16_Batch'),
    HardTanh(name='L17_HardTanh'),

    BinarizedSpatialConvolution(512,3,3, padding='SAME', bias=False,name='L18_Convolution'),
    SpatialMaxPooling(2,2,2,2,name='L19_MaxPooling'),
    BatchNormalization(name='L20_Batch'),
    HardTanh(name='L21_HardTanh'),

    BinarizedAffine(1024, bias=False,name='L22_FullyConnected'),
    BatchNormalization(name='L23_Batch'),
    HardTanh(name='L24_HardTanh'),

    BinarizedAffine(1024, bias=False,name='L25_FullyConnected'),
    BatchNormalization(name='L26_Batch'),
    HardTanh(name='L27_HardTanh'),

    BinarizedAffine(10,bias=False,name='L28_FullyConnected'),
    BatchNormalization(name='L29_Batch')
])
