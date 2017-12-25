from nnUtils import *

model = Sequential([
    SpatialConvolution(64,3,3,1,1, padding='SAME', bias=False),
    ReLU(),
    SpatialMaxPooling(2,2,2,2,name='L5_MaxPooling'),

    SpatialConvolution(128,3,3,1,1, padding='SAME', bias=False),
    ReLU(),
    SpatialMaxPooling(2,2,2,2,name='L5_MaxPooling'),

    SpatialConvolution(256,3,3,1,1, padding='SAME', bias=False),
    ReLU(),
    SpatialConvolution(256, 3, 3,1,1, padding='SAME', bias=False),
    ReLU(),
    SpatialMaxPooling(2,2,2,2,name='L12_MaxPooling'),

    SpatialConvolution(512,3,3, padding='SAME', bias=False),
    ReLU(),
    SpatialConvolution(512,3,3, padding='SAME', bias=False),
    ReLU(),
    SpatialMaxPooling(2,2,2,2,name='L19_MaxPooling'),

    SpatialConvolution(512, 3, 3, padding='SAME', bias=False),
    ReLU(),
    SpatialConvolution(512, 3, 3, padding='SAME', bias=False),
    ReLU(),
    SpatialMaxPooling(2, 2, 2, 2, name='L19_MaxPooling'),

    Affine(1024, bias=False),
    ReLU(),
    Affine(1024, bias=False),
    ReLU(),
    Affine(10, bias=False)
])
