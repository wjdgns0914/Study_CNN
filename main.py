import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
# from progress.bar import Bar
from datetime import datetime
from tensorflow.python.platform import gfile
from data import *
from evaluate import evaluate
from tqdm import tqdm
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
timestr = '-'.join(str(x) for x in list(tuple(datetime.now().timetuple())[:6]))
MOVING_AVERAGE_DECAY = 0.997
FLAGS = tf.app.flags.FLAGS
tf.set_random_seed(777)  # reproducibility

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 80,
                            """Number of epochs to train. -1 for unlimited""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_string('model', 'BNN_MNIST0_nodrop',  #BNN_MNIST0_nodrop
                           """Name of loaded model.""")
tf.app.flags.DEFINE_string('save', timestr+"HardTanh_Var_Drift",
                           """Name of saved dir.""")
tf.app.flags.DEFINE_string('load', None,
                           """Name of loaded dir.""")
tf.app.flags.DEFINE_string('dataset', 'MNIST',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_string('summary', True,
                           """Record summary.""")
tf.app.flags.DEFINE_string('Drift1', True,
                           """Drift or Not.""")
tf.app.flags.DEFINE_string('Drift2', True,
                           """Drift or Not.""")
tf.app.flags.DEFINE_string('Variation', True,
                           """Variation or Not.""")
tf.app.flags.DEFINE_string('log', 'ERROR',
                           'The threshold for what messages will be logged '
                            """DEBUG, INFO, WARN, ERROR, or FATAL.""")

FLAGS.checkpoint_dir = './results/' + FLAGS.save
FLAGS.log_dir = FLAGS.checkpoint_dir + '/log/'
# tf.logging.set_verbosity(FLAGS.log)

##파라미터 개수를 세는 용도의 함수
def count_params(var_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
    return num

##Writer에 작성 할 내용을 정의하는 함수
def add_summaries(scalar_list=[], activation_list=[], var_list=[], grad_list=[],Wbin_list=[],Wfluc_list=[]):

    for var in scalar_list:
        if var is not None:
            tf.summary.scalar(var.op.name, var)

    for grad, var in grad_list:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    for var in var_list:
        if var is not None:
            ful = var.op.name
            tf.summary.histogram(ful.split('/')[0]+'/0/'+ful.split('/')[1], var)
            sz = var.get_shape().as_list()
            if len(sz) == 4 and sz[2] == 3:
                kernels = tf.transpose(var, [3, 0, 1, 2])
                tf.summary.image(var.op.name + '/kernels',group_batch_images(kernels), max_outputs=1)
    for var in var_list:
        if var is not None:
            # index=np.array([1]*(len(var.get_shape().as_list())-2)+[0]+[1])

            ful = var.op.name  # full name
            for i in range(2):
                index=[np.random.randint(j) for j in var.get_shape().as_list()]
                tf.summary.scalar(ful.split('/')[0]+'/W'+str(index)+'/0/'+ful.split('/')[1], var[index])

    for activation in activation_list:
        if activation is not None:
            ful=activation.op.name
            tf.summary.histogram(ful.split('/')[0]+'/2/'+'activations', activation)

    for Wbin in Wbin_list:
        if Wbin is not None:
            ful=Wbin.op.name  #full name
            tf.summary.histogram(ful.split('/')[0]+'/1/'+ful.split('/')[1], Wbin)
    for Wbin in Wbin_list:
        if Wbin is not None:
            # index = np.array([1] * (len(Wbin.get_shape().as_list()) - 2) + [0] + [1])
            ful = Wbin.op.name  # full name
            for i in range(2):
                index = [np.random.randint(j) for j in Wbin.get_shape().as_list()]
                tf.summary.scalar(ful.split('/')[0]+'/W'+str(index)+'/1/'+ful.split('/')[1], Wbin[index])


    for Wfluc in Wfluc_list:
        if Wfluc is not None:
            # index = np.array([1] * (len(Wfluc.get_shape().as_list()) - 2) + [0] + [1])
            ful = Wfluc.op.name  # full name
            for i in range(2):
                index = [np.random.randint(j) for j in Wfluc.get_shape().as_list()]
                tf.summary.scalar(ful.split('/')[0] + '/W' + str(index) + '/2/' + ful.split('/')[1],
                                  Wfluc[index])
            #tf.summary.scalar(activation.op.name + '/sparsity', tf.nn.zero_fraction(activation))

##LR을 decay시켜주는 함수
def _learning_rate_decay_fn(learning_rate, global_step):
    print("learning_rate_decay_fn is executed!")
    return tf.train.exponential_decay(
      learning_rate,
      global_step,
      decay_steps=1000,
      decay_rate=0.9,
      staircase=True)
learning_rate_decay_fn = _learning_rate_decay_fn

## model을 data로 training 시켜주는 함수
def train(model, data,
          batch_size=128,
          learning_rate=FLAGS.learning_rate,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=-1):
    # tf Graph input

    with tf.name_scope('data'):
        x, yt = data.next_batch(batch_size)
    global_step =  tf.get_variable('global_step', shape=[],dtype=tf.int64,
                         initializer=tf.constant_initializer(0),
                         trainable=False)
    tf.add_to_collection("Step",global_step)

    # use_for_coming_batch = [{}, {}]  # 순서대로 Wbin,Wfluc
    # tf.add_to_collection('use_for_coming_batch', use_for_coming_batch)
    y = model(x, is_training=True)
    # Define loss and optimizer
    with tf.name_scope('objective'):
        yt_one=tf.one_hot(yt,10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yt_one, logits=y))
        accuracy=tf.reduce_mean(tf.cast(tf.equal(yt, tf.cast(tf.argmax(y, dimension=1),dtype=tf.int32)),dtype=tf.float32))
        # accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
    opt= tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    opt = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, 'Adam',
                                          gradient_noise_scale=None, gradient_multipliers=None,
                                          clip_gradients=None, #moving_average_decay=0.9,
                                           update_ops=None, variables=None, name=None)  #learning_rate_decay_fn=learning_rate_decay_fn,
        #grads = opt.compute_gradients(loss)
        #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    print("Definite Moving Average...")
    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy] + tf.trainable_variables())
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    tf.summary.scalar('loss/training', loss_avg)
    accuracy_avg = ema.average(accuracy)
    tf.summary.scalar('accuracy/training', accuracy_avg)

    check_loss = tf.\
        check_numerics(loss, 'model diverged: loss->nan')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([opt]):
        train_op = tf.group(*updates_collection)
    print("Make summary for writer...")
    list_W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='L')
    list_Wbin = tf.get_collection('Binarized_Weight', scope='L')
    list_Wfluc = tf.get_collection('Fluctuated_Weight', scope='L')

    list_pre_Wbin = tf.get_collection('pre_Wbin', scope='L')
    list_pre_Wfluc = tf.get_collection('pre_Wfluc', scope='L')

    list_pre_Wbin_op = tf.get_collection('pre_Wbin_update_op', scope='L')
    list_pre_Wfluc_op = tf.get_collection('pre_Wfluc_update_op', scope='L')

    if FLAGS.summary:
        add_summaries(scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
            activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
            var_list=list_W,Wbin_list=list_Wbin,Wfluc_list=list_Wfluc)
            # grad_list=grads)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    print("Open Session...")
    # Configure options for session
    gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.9)
    sess = tf.InteractiveSession(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_batches = int(data.size[0] / batch_size)
    print("Check the collections...")
    print("list_W:\n",list_W,'\nNum:',len(list_W))
    print("list_Wbin:\n:",list_Wbin,'\nNum:',len(list_Wbin))
    print("list_Wfluc:\n:",list_Wfluc,'\nNum:',len(list_Wfluc))
    # print("list_pre_Wbin:\n:", list_pre_Wbin, '\nNum:', len(list_pre_Wbin))
    # print("list_pre_Wfluc:\n:", list_pre_Wfluc, '\nNum:', len(list_pre_Wfluc))
    print("list_pre_Wbin_op:\n:", list_pre_Wbin_op, '\nNum:', len(list_pre_Wbin_op))
    print("list_pre_Wfluc_op:\n:", list_pre_Wfluc_op, '\nNum:', len(list_pre_Wfluc_op))

    print('We start training..num of trainable paramaters: %d' %count_params(tf.trainable_variables()))
    best_acc=0
    file = open(FLAGS.checkpoint_dir + "/model.py", "a")
    print("Drift1:",FLAGS.Drift1,"\nDrift2",FLAGS.Drift2,"\nVariation",FLAGS.Variation,file=file)
    print("\nLR:", FLAGS.learning_rate, "\nbatch_size", FLAGS.batch_size, "\nDataset", FLAGS.dataset, file=file)

    for i in range(num_epochs):
        # file = open(FLAGS.checkpoint_dir + "/model.py", "a")
        print('"""', file=file)
        print('Started epoch %d' % (i+1))
        print('Started epoch %d' % (i + 1),file=file)
        count_num=np.array([0,0,0,0,0,0,0,0,0,0])
        for j in tqdm(range(num_batches)):
            #tf.add_to_collection('use_for_this_batch', save_for_next_batch)    #이 코드가 메모리 에러를 일으키는 주범이었다. dict은 알아서 메모리 관리 잘하고 있었음.
            # print("drift_factor=",sess.run(tf.get_collection("testt")))
            list_run = sess.run(list_Wbin+list_Wfluc+[train_op, loss]+[y,yt])  #train_op를 통해 업데이트를 하기 전에 list_Wbin,Wfluc에 있는 var들의 값을 save for next batch
            # if j>38 and j<40:
            #     print("prediction:",list_run[-2],"  correct:",list_run[-1])
            # if i==2:
            #     aa = list_run[-2][0]
            #     plt.imshow(list_run[-2][0].reshape([28, 28]))
            #     plt.show()
            #     print(list_run[-1][0])
            #     plt.imshow(list_run[-2][3].reshape([28, 28]))
            #     plt.show()
            #     print(list_run[-1][3])
            unique_elements,elements_counts=np.unique(list_run[-1],return_counts=True)
            num_set=dict(zip(unique_elements,elements_counts))
            for ii in range(10):
                if num_set.__contains__(ii):
                    count_num[ii]=count_num[ii]+num_set[ii]
            if FLAGS.Variation:
                for index, value in enumerate(list_run[0:len(list_Wbin)]):
                    sess.run(list_pre_Wbin_op[index],{list_pre_Wbin[index]:value})
                for index, value in enumerate(list_run[len(list_Wbin):len(list_Wbin + list_Wfluc)]):
                    sess.run(list_pre_Wfluc_op[index],{list_pre_Wfluc[index]:value})
            if j%10==0:
                summary_writer.add_summary(sess.run(summary_op), global_step=sess.run(global_step))

        step, acc_value, loss_value, summary = sess.run([global_step, accuracy_avg, loss_avg, summary_op])
        """
        20171204:avg기능을 빼고 코드를 돌려보니까 training set에서의 정확도가 98퍼가 되었다가 96퍼가 되었다가 왔다갔다한다.
        이유: 위에서 accuracy 를 돌릴 때 데이터가 64개밖에 안쓰인다, 그래서 정확도의 격차가 좀 있었던 것
        """
        # temp0, temp1 = sess.run([x, yt])
        # aa = temp0[0]
        # plt.imshow(temp0[0].reshape([28, 28]))
        # plt.show()
        # print(temp1[0])
        # plt.imshow(temp0[3].reshape([28, 28]))
        # plt.show()
        # print(temp1[3])
        print(["%d : "%i+str(count_num[i]) for i in range(10)]," Totral num: ",count_num.sum())
        print(["%d : " % i + str(count_num[i]) for i in range(10)], " Totral num: ", count_num.sum(),file=file)
        print('Training - Accuracy: %.3f' % acc_value,'  Loss:%.3f'% loss_value)
        print('Training - Accuracy: %.3f' % acc_value, '  Loss:%.3f' % loss_value,file=file)

        saver.save(sess, save_path=checkpoint_dir + '/model.ckpt', global_step=global_step)
        test_acc, test_loss = evaluate(model, FLAGS.dataset,
                                       checkpoint_dir=checkpoint_dir)
        # log_dir=log_dir)
        print('Test     - Accuracy: %.3f' % test_acc, '  Loss:%.3f' % test_loss)
        print('Test     - Accuracy: %.3f' % test_acc, '  Loss:%.3f' % test_loss,file=file)
        if best_acc<test_acc:
            best_acc=test_acc
            saver.save(sess, save_path=checkpoint_dir + '/best_model.ckpt', global_step=global_step)
        print('Best     - Accuracy: %.3f' % best_acc)
        print('Best     - Accuracy: %.3f' % best_acc,file=file)
        summary_out = tf.Summary()
        summary_out.ParseFromString(summary)
        summary_out.value.add(tag='accuracy/test', simple_value=test_acc)
        summary_out.value.add(tag='loss/test', simple_value=test_loss)
        summary_writer.add_summary(summary_out, step)
        summary_writer.flush()
        print('"""', file=file)


    # When done, ask the threads to stop.
    file.close()
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    summary_writer.close()
"""
설명2:
1)what is the argv
Argv in Python
The list of command line arguments passed to a Python script. argv[0] is the script name (it is operating system 
dependent whether this is a full pathname or not). If the command was executed using the -c command line option 
to the interpreter, argv[0] is set to the string '-c'.

지금은 FLAGS가 글로벌하게 선언이 되어있어서 argv가 전달된다는게 큰의미는 없다.
(전달안되도 어차피 글로벌이라 그냥 사용가능.tf.app.run()은 그냥 one line fast argument parser로 생각하면 될 듯 하다.)
"""

def main(argv=None):  # pylint: disable=unused-argument
    print(argv)
    """
    설명3:
    1) gfile은 약간 폴더,파일쪽 다루는 패키지인가보다, 만약 checkpoint_dir에서 지정 된 폴더가 없으면 그걸 만들어서
    2) os.path.join은 단순히 경로 만들어주는 함수다, 저절로 / 를 추가해주는게 편리한 점
    3) assert FLAGS.model로 명시한 모델이 있으면 통과 없으면 뒤에 있는 오류메시지 발생
    4) 해당 파이썬 파일을(인풋1) 인풋2로 복사한다.
    5) 해당 모델을 import한다, importlib.import_module()함수는 코드 과정 중에 패키지를 import 할 때 쓰는 듯 하다.
    6) data = get_data_provider(FLAGS.dataset, training=True) 에서 training에 따라 trainset 혹은 testset을 불러온다.
    7) 위에서 정의한 train함수를 실행 - train을 진행하기 전에 data.py파일을 살펴보자
    """
    if not gfile.Exists(FLAGS.checkpoint_dir):
        # gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        gfile.MakeDirs(FLAGS.checkpoint_dir)
        model_file = os.path.join('models', FLAGS.model + '.py')
        assert gfile.Exists(model_file), 'no model file named: ' + model_file
        gfile.Copy(model_file, FLAGS.checkpoint_dir + '/model.py')
    m = importlib.import_module('models.' + FLAGS.model)
    # if FLAGS.dataset=="MNIST":
    #     from tensorflow.examples.tutorials.mnist import input_data
    #     mnist = input_data.read_data_sets("Datasets/MNIST", one_hot=False)
    # else:
    #     mnist = None
    data = get_data_provider(FLAGS.dataset, training=True)

    train(m.model, data,
          batch_size=FLAGS.batch_size,
          checkpoint_dir=FLAGS.checkpoint_dir,
          log_dir=FLAGS.log_dir,
          num_epochs=FLAGS.num_epochs)

"""

설명1:여기서부터 설명한다,일단 이 코드를 실행하면 위에서부터 라인바이라인 실행이 된다.
다만 함수는 라인바이라인 실행 되는 것이 아니라 함수가 있다는 것을 알려줄 수 있는 '선언'만 하고 넘어가게 된다.
그러면 마지막에 실행되는 것이 아래의 조건문이다.
모든 파일은 실행 될 때 __name__이 부여되는 듯 하다. __main__이라는건 이 파일이! 모듈 같이 간접적으로 실행되었다는게 아니라
직접적으로 실행되었다는 뜻이다, 즉 아래의 코드는 이 파일이 직접 실행 된건지, 아니면 다른 코드에 종속적으로 실행된건지를 판단한다.
-여기서 종속적으로 실행되면 __name__=? 인지는 아직 모른다.
이걸 판단해서 직접적으로 실행된거면 tf.app.run()을 실행한다, 아니면 아무것도 실행안된다, 즉 아무것도 안된다.
이는 이 파일이 부정할 수 없는 메인파일이라는 것을 말해준다.

여기서 처음에 들었던 의문-tf.app.run()은 argv를 받아서 main()이라는 이름의 함수(default)에 전달해서 실행시킨다고 하는데
왜 바로 아래 판단문에 그 함수의 내용을 쓰지 않는걸까? 굳이 한번의 매개를 거치는 이유가 무엇인가?
->답변: 우리는 FLAGS를 이용해서 커맨드창에서 파라미터설정을 입력 받는다, 그리고 그걸 wraping해서 함수에 쓰는거다.
tf.app.run()은 그걸 wraping해주는 좋은 함수이고, 그걸 main()함수에 전달하도록 되어있다. 즉 tf.app.run()함수 자체가 main()함수를
필요로 하는 것이다.
#Runs the program with an optional 'main' function and 'argv' list.
->그렇다면 질문은 tf.app.run()가 왜 이렇게 쓰이냐!로 되는데 그건 나중에 생각해보자.
추측: 만약 이 함수를 안쓰면 우리는 string인 cmd 명령어를 쪼개서, 그 중에 어떤 부분이 파일에 정의되어있는지 체크해서
그걸 argv로 만들어줘야한다. 
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)        #https://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
보통 위처럼 세줄정도는 나올텐데, 그것보다는 한줄로 써서 위의 기능들을 취하는 것 아닐까?
그리고 main()으로 구분하면 가독성도 좋아지고, __main__이 아니어도 어떻게 접근이 가능할수도 있고..여러가지 가능성이 나오게되는 장점도 있다

"""
if __name__ == '__main__':
    # print(callable(main))
    # print(locals())
    tf.app.run()


"""
Started epoch 1
100%|██████████| 859/859 [00:26<00:00, 32.33it/s]
['0 : 5465', '1 : 6181', '2 : 5435', '3 : 5592', '4 : 5363', '5 : 4967', '6 : 5426', '7 : 5742', '8 : 5357', '9 : 5448']  Totral num:  54976
Training - Accuracy: 0.979   Loss:0.061
Test     - Accuracy: 0.982   Loss:0.056

Started epoch 2
100%|██████████| 859/859 [00:24<00:00, 34.39it/s]
['0 : 5360', '1 : 6209', '2 : 5491', '3 : 5671', '4 : 5333', '5 : 4969', '6 : 5471', '7 : 5688', '8 : 5321', '9 : 5463']  Totral num:  54976
Training - Accuracy: 0.984   Loss:0.049
Test     - Accuracy: 0.986   Loss:0.048
Best     - Accuracy: 0.986

Started epoch 3
100%|██████████| 859/859 [00:24<00:00, 34.49it/s]
['0 : 5431', '1 : 6201', '2 : 5457', '3 : 5579', '4 : 5325', '5 : 4953', '6 : 5384', '7 : 5761', '8 : 5402', '9 : 5483']  Totral num:  54976
Training - Accuracy: 0.988   Loss:0.038
Test     - Accuracy: 0.989   Loss:0.038
Best     - Accuracy: 0.989

Started epoch 4
100%|██████████| 859/859 [00:24<00:00, 34.69it/s]
['0 : 5435', '1 : 6217', '2 : 5426', '3 : 5638', '4 : 5361', '5 : 4973', '6 : 5461', '7 : 5728', '8 : 5269', '9 : 5468']  Totral num:  54976
Training - Accuracy: 0.990   Loss:0.031
Test     - Accuracy: 0.982   Loss:0.060
Best     - Accuracy: 0.989

Started epoch 5
100%|██████████| 859/859 [00:24<00:00, 35.39it/s]
['0 : 5428', '1 : 6196', '2 : 5431', '3 : 5618', '4 : 5380', '5 : 4950', '6 : 5415', '7 : 5730', '8 : 5399', '9 : 5429']  Totral num:  54976
Training - Accuracy: 0.990   Loss:0.031
Test     - Accuracy: 0.988   Loss:0.047
Best     - Accuracy: 0.989

Started epoch 6
100%|██████████| 859/859 [00:24<00:00, 35.12it/s]
['0 : 5458', '1 : 6147', '2 : 5466', '3 : 5636', '4 : 5343', '5 : 4974', '6 : 5396', '7 : 5763', '8 : 5360', '9 : 5433']  Totral num:  54976
Training - Accuracy: 0.992   Loss:0.026
Test     - Accuracy: 0.989   Loss:0.038
Best     - Accuracy: 0.989

Started epoch 7
100%|██████████| 859/859 [00:24<00:00, 34.86it/s]
['0 : 5412', '1 : 6097', '2 : 5494', '3 : 5635', '4 : 5375', '5 : 4956', '6 : 5417', '7 : 5758', '8 : 5360', '9 : 5472']  Totral num:  54976
Training - Accuracy: 0.993   Loss:0.024
Test     - Accuracy: 0.988   Loss:0.044
Best     - Accuracy: 0.989

Started epoch 8
100%|██████████| 859/859 [00:24<00:00, 35.05it/s]
['0 : 5429', '1 : 6232', '2 : 5414', '3 : 5593', '4 : 5366', '5 : 4958', '6 : 5422', '7 : 5732', '8 : 5373', '9 : 5457']  Totral num:  54976
Training - Accuracy: 0.994   Loss:0.020
Test     - Accuracy: 0.989   Loss:0.051
Best     - Accuracy: 0.989

Started epoch 9
100%|██████████| 859/859 [00:25<00:00, 34.20it/s]
['0 : 5447', '1 : 6131', '2 : 5459', '3 : 5604', '4 : 5335', '5 : 4987', '6 : 5432', '7 : 5774', '8 : 5354', '9 : 5453']  Totral num:  54976
Training - Accuracy: 0.994   Loss:0.019
Test     - Accuracy: 0.989   Loss:0.042
Best     - Accuracy: 0.989

Started epoch 10
100%|██████████| 859/859 [00:24<00:00, 35.30it/s]
['0 : 5411', '1 : 6163', '2 : 5490', '3 : 5622', '4 : 5351', '5 : 4931', '6 : 5447', '7 : 5748', '8 : 5391', '9 : 5422']  Totral num:  54976
Training - Accuracy: 0.994   Loss:0.021
Test     - Accuracy: 0.989   Loss:0.047
Best     - Accuracy: 0.989

"""

