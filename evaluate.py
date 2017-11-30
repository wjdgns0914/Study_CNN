from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from data import get_data_provider

def evaluate(model, dataset,
        batch_size=128,
        checkpoint_dir='./checkpoint',sess=tf.Session()):
    with tf.Graph().as_default() as g:
        data = get_data_provider(dataset, training=False)

        x, yt = data.generate_batches(batch_size)
        is_training = tf.placeholder(tf.bool,[],name='is_training')
        # Build the Graph that computes the logits predictions
        y = model(x, is_training=False)

        # Calculate predictions.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
        accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y,yt,1), tf.float32))

        # Restore the moving average version of the learned variables for eval.
        #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()#variables_to_restore
        # Configure options for session
        sess = tf.Session(
                config=tf.ConfigProto(
                            log_device_placement=False,
                            allow_soft_placement=True,
                            )
                        )
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'/')
        if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

         # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
             threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))

            num_batches = int(math.ceil(data.size[0] / batch_size))
            total_acc = 0  # Counts the number of correct predictions per batch.
            total_loss = 0 # Sum the loss of predictions per batch.
            step = 0
            while step < num_batches and not coord.should_stop():
              acc_val, loss_val = sess.run([accuracy, loss])
              total_acc += acc_val
              total_loss += loss_val
              step += 1

            # Compute precision and loss
            total_acc /= num_batches
            total_loss /= num_batches


        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)   #이 녀석 때문에 sess를 하나 더 생성해야한다. 그냥 sess를 인자로 받아오면
        #조정자가 충돌하는거 같다
        #아니네, 아래를 보니까 애를 독립시킨 진정한 이유는 저장 된 웨이트를 통해 evaluation을 할 수 있게하려고인거 같다.
        return total_acc, total_loss

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('checkpoint_dir', './results/model',
                             """Directory where to read model checkpoints.""")
  tf.app.flags.DEFINE_string('dataset', 'cifar10',
                             """Name of dataset used.""")
  tf.app.flags.DEFINE_string('model_name', 'model',
                             """Name of loaded model.""")
  tf.app.flags.DEFINE_string('Drift', False,
                             """Drift or Not.""")

  FLAGS.log_dir = FLAGS.checkpoint_dir+'/log/'
      # Build the summary operation based on the TF collection of Summaries.
      # summary_op = tf.merge_all_summaries()

      # summary_writer = tf.train.SummaryWriter(log_dir)
          # summary = tf.Summary()
          # summary.ParseFromString(sess.run(summary_op))
          # summary.value.add(tag='accuracy/test', simple_value=precision)
          # summary_writer.add_summary(summary, global_step)

  tf.app.run()