import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.contrib.framework import get_name_scope
from numpy import zeros
from numpy import ones
FLAGS = tf.app.flags.FLAGS
print(FLAGS.Drift)
print(FLAGS.Variation)

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with tf.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,-1,1)
            return tf.sign(x)
if FLAGS.Variation==True:
    Reset_Meanmean, Reset_Meanstd = 0., 0.1707
    Reset_Stdmean, Reset_Stdstd = 0.0942, 0.01884
    Set_Meanmean, Set_Meanstd = 0., 0.1538
    Set_Stdmean, Set_Stdstd = 0.1311, 0.06894
else:
    Reset_Meanmean, Reset_Meanstd = 0., 0.0000000001
    Reset_Stdmean, Reset_Stdstd = 0., 0.
    Set_Meanmean, Set_Meanstd = 0., 0.0000000001
    Set_Stdmean, Set_Stdstd = 0., 0.

@tf.RegisterGradient("fluc_grad")
def fluc_grad(op,grad):
    shape=op.inputs[1]._shape_as_list()
    return grad,tf.zeros(shape=shape)

def fluctuate(x,scale=1):
    filter_shape = x.get_shape().as_list()
    g = tf.get_default_graph()
    pre_Wbin = tf.Variable(initial_value=tf.zeros(shape=filter_shape),name='pre_Wbin',trainable=False)
    pre_Wbin_val_place=tf.placeholder(dtype=tf.float32,shape=filter_shape)
    pre_Wbin_update_op=pre_Wbin.assign(pre_Wbin_val_place)
    pre_Wfluc = tf.Variable(initial_value=tf.zeros(shape=filter_shape),name='pre_Wfluc',trainable=False)
    pre_Wfluc_val_place = tf.placeholder(dtype=tf.float32, shape=filter_shape)
    pre_Wfluc_update_op = pre_Wfluc.assign(pre_Wfluc_val_place)

    tf.add_to_collection('pre_Wbin',pre_Wbin_val_place)
    tf.add_to_collection('pre_Wbin_update_op', pre_Wbin_update_op)
    tf.add_to_collection('pre_Wfluc', pre_Wfluc_val_place)
    tf.add_to_collection('pre_Wfluc_update_op', pre_Wfluc_update_op)
    # if tf.get_collection('use_for_coming_batch')==[]:
    #     pre_Wbin=pre_Wfluc=tf.zeros(shape=filter_shape)
    # else:
    #     pre_Wbin=tf.get_collection('use_for_coming_batch')[0][x.name.split('/')[0]]
    #     pre_Wfluc=tf.get_collection('use_for_coming_batch')[1][x.name.split('/')[0]]
        # g = tf.get_default_graph()

    with tf.name_scope("Fluctuated") as name:
        Reset_Meanvalue = tf.Variable(tf.random_normal(shape=filter_shape,
                                            mean=Reset_Meanmean,
                                            stddev=Reset_Meanstd, dtype=tf.float32),
                                            name="Reset_MeanValue",trainable=False)
        Reset_Stdvalue = tf.Variable(((Reset_Meanvalue - Reset_Meanmean) / Reset_Meanstd)
                                     * Reset_Stdstd + Reset_Stdmean,
                                     name='Reset_Stdvale',
                                     trainable=False)
        Set_Meanvalue = tf.Variable(tf.random_normal(shape=filter_shape,
                                        mean=Set_Meanmean,
                                        stddev=Set_Meanstd,
                                        dtype=tf.float32),
                                        name="Set_MeanValue",trainable=False)
        Set_Stdvalue = tf.Variable(((Set_Meanvalue - Set_Meanmean) / Set_Meanstd)
                                   * Set_Stdstd + Set_Stdmean,
                                   name='Set_Stdvalue',
                                   trainable=False)
        fluc_Reset = tf.reshape(tf.distributions.Normal(loc=Reset_Meanvalue, scale=Reset_Stdvalue).sample(1),filter_shape)
        fluc_Set = tf.reshape(tf.distributions.Normal(loc=Set_Meanvalue, scale=Set_Stdvalue).sample(1), filter_shape)
        with g.gradient_override_map({"Mul": "fluc_grad","Cast": "Identity",
                                      "Equal": "fluc_grad","Greater":"fluc_grad",
                                      "LessEqual":"fluc_grad","NotEqual":"fluc_grad",
                                      "Add": "fluc_grad"}):
            # assign 1 to elements which have same state with pre-state
            keep_element = tf.cast(tf.equal(x,pre_Wbin), tf.float32)
            tf.add_to_collection('test',keep_element)
            # assign 1 to elements which have different state with pre-state
            update_element = tf.cast(tf.not_equal(x,pre_Wbin), tf.float32)
            # 이 부분에서 쓰지도않는 랜덤 값이 많이 발생하는데 일단 두고 나중에 고치든가 하자
            Wfluc_Reset = update_element * fluc_Reset * tf.cast(x > 0, tf.float32)
            Wfluc_Set = update_element * fluc_Set * tf.cast(x <= 0, tf.float32)
            # fluctuation이 적용 된 최종 weight 값,Reset,set에 drift를 따로 적용하기 위해 pre_Wbin이 0보다 큰 부분,작은 부분, 두 부분으로 나누었다.
            step_col=tf.get_collection("Step")
            if FLAGS.Drift and step_col!=[]:
                batch_num = step_col[0]
                drift_factor =tf.cast((1 + batch_num) / batch_num,dtype=tf.float32)
                drift_scale = tf.cond(tf.equal(batch_num, 0), lambda: tf.constant(0.),     #tf.cast(tf.equal(batch_num, 0), dtype=tf.float32)
                                      lambda: tf.cast(0.09 * tf.log(drift_factor) / tf.log(tf.constant(10, dtype=tf.float32)), dtype=tf.float32))
            else:
                drift_scale=tf.constant(0.)
            with tf.control_dependencies([drift_scale]):
                Wfluc = tf.multiply(x, update_element) +tf.cast(tf.greater(pre_Wbin,0), tf.float32) * keep_element * (pre_Wfluc + drift_scale)+ \
                        tf.cast(tf.less_equal(pre_Wbin,0), tf.float32) * keep_element * pre_Wfluc * 1. \
                        + Wfluc_Reset + Wfluc_Set

        return Wfluc


def BinarizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedSpatialConvolution'):
    def b_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name,reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bin_x = binarize(x)
            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(bin_x, fluc_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv2d

def BinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())

            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)

            out = tf.nn.conv2d(x, fluc_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, reuse=None, name='SpatialConvolution'):
    def conv2d(x, is_training=True):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name, reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d
#Fully connected layer
def Affine(nOutputPlane, bias=True, name=None, reuse=None):
    def affineLayer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=name, default_name='Affine', reuse=reuse):
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            output = tf.matmul(reshaped, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return affineLayer

def BinarizedAffine(nOutputPlane, bias=True, name=None, reuse=None):
    def b_affineLayer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=name, default_name='Affine', reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = binarize(x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)

            output = tf.matmul(reshaped, fluc_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return b_affineLayer

def BinarizedWeightOnlyAffine(nOutputPlane, bias=True, name=None, reuse=None):
    def bwo_affineLayer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=name, default_name='Affine', reuse=reuse):

            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())
            bin_w = binarize(w)
            fluc_w = fluctuate(bin_w)
            tf.add_to_collection('Binarized_Weight', bin_w)
            tf.add_to_collection('Fluctuated_Weight', fluc_w)

            output = tf.matmul(reshaped, fluc_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return bwo_affineLayer
#bias 더 해주는 레이어
def Linear(nInputPlane, nOutputPlane):
    return Affine(nInputPlane, nOutputPlane, add_bias=False)


def wrapNN(f,name,*args,**kwargs):
    def layer(x, scope=name, is_training=True):
        return f(x,scope=scope,*args,**kwargs)
    return layer

def Dropout(p, name='Dropout'):
    def dropout_layer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            # def drop(): return tf.nn.dropout(x,p)
            # def no_drop(): return x
            # return tf.cond(is_training, drop, no_drop)
            if is_training:
                return tf.nn.dropout(x,p)
            else:
                return x
    return dropout_layer

def ReLU(name='ReLU'):
    def layer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            return tf.nn.relu(x)
    return layer

def HardTanh(name='HardTanh'):
    def layer(x, is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            return tf.clip_by_value(x,-1,1)
    return layer


def View(shape, name='View'):
    return wrapNN(tf.reshape,shape=shape)

def SpatialMaxPooling(kW, kH=None, dW=None, dH=None, padding='VALID',
            name='SpatialMaxPooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def max_pool(x,is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
              return tf.nn.max_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return max_pool

def SpatialAveragePooling(kW, kH=None, dW=None, dH=None, padding='VALID',
        name='SpatialAveragePooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def avg_pool(x,is_training=True):
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
              return tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return avg_pool

def BatchNormalization(name='BatchNormalization',*kargs, **kwargs):
    output=wrapNN(tf.contrib.layers.batch_norm,name, *kargs, **kwargs)
    return output


def Sequential(moduleList):
    def model(x, is_training=True):
    # Create model
        output = x
        #with tf.variable_op_scope([x], None, name):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
    return model

def Concat(moduleList, dim=3):
    def model(x, is_training=True):
    # Create model
        outputs = []
        for i,m in enumerate(moduleList):
            name = 'layer_'+str(i)
            with tf.variable_scope(values=[x], name_or_scope=name, default_name='Layer'):
                outputs[i] = m(x, is_training=is_training)
            output = tf.concat(dim, outputs)
        return output
    return model

def Residual(moduleList, name='Residual'):
    m = Sequential(moduleList)
    def model(x, is_training=True):
    # Create model
        with tf.variable_scope(values=[x], name_or_scope=None, default_name=name):
            output = tf.add(m(x, is_training=is_training), x)
            return output
    return model