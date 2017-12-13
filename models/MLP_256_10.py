from nnUtils import *
FLAGS = tf.app.flags.FLAGS
Dri1=FLAGS.Drift1
Dri2=FLAGS.Drift2
model = Sequential([
    BinarizedWeightOnlySpatialConvolution(64,3,3,padding='SAME',bias=False,name='L0_Conv',Drift=Dri1),
    BinarizedWeightOnlyAffine(256, bias=False,name='L1_FullyConnected',Drift=Dri2),
    # BatchNormalization(name='L17_Batch'),
    ReLU(name='L2_ReLU'),
    BinarizedWeightOnlyAffine(10,bias=False,name='L3_FullyConnected',Drift=Dri2),
])
