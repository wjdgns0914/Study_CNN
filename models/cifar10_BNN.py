from nnUtils import *
from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(64,3,3,1,1, padding='SAME', bias=False, name='L1_Conv'),
    BatchNormalization(name='L1_Batch'),
    ReLU(),
    BinarizedWeightOnlySpatialConvolution(64,3,3,1,1, padding='SAME', bias=False, name='L2_Conv'),
    BatchNormalization(name='L2_Batch'),
    ReLU(),
    SpatialMaxPooling(2,2,2,2,name='L3_MaxPooling'),

    BinarizedWeightOnlySpatialConvolution(128,3,3,1,1, padding='SAME', bias=False, name='L4_Conv'),
    BatchNormalization(name='L4_Batch'),
    ReLU(),
    BinarizedWeightOnlySpatialConvolution(128, 3, 3,1,1, padding='SAME', bias=False,name='L5_Conv'),
    BatchNormalization(name='L5_Batch'),
    ReLU(),
    SpatialMaxPooling(2,2,2,2,name='L6_MaxPooling'),

    BinarizedAffine(512, bias=False,name='L7_Affine'),
    BatchNormalization(name='L7_Batch'),
    ReLU(),
    Dropout(0.6),
    BinarizedAffine(256, bias=False,name='L8_Affine'),
    BatchNormalization(name='L8_Batch'),
    ReLU(),
    Dropout(0.6),
    BinarizedAffine(10, bias=False,name='L9_Affine')

    # # SpatialConvolution(512,3,3, padding='SAME', bias=False, name='L7_Conv'),
    # # ReLU(),
    # # SpatialConvolution(512,3,3, padding='SAME', bias=False, name='L8_Conv'),
    # # ReLU(),
    # # SpatialMaxPooling(2,2,2,2,name='L9_MaxPooling'),
    #
    # Affine(512, bias=False,name='L10_Affine'),
    # ReLU(),
    # Affine(256, bias=False,name='L11_Affine'),
    # ReLU(),
    # Affine(10, bias=False,name='L12_Affine')
])
