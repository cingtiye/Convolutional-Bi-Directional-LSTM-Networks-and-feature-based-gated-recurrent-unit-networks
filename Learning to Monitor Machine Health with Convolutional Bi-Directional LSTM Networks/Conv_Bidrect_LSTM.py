# -*- coding: utf-8

import numpy as np
import pickle
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

class CBLSTM(object):
    def __init__(self,
                 MODEL_TYPE     = 'Regression',
                 FILTER_NUM     = 150,
                 FILTER_SIZE    = 10,
                 POOL_SIZE      = 5,
                 INPUT_LENGTH   = 12,
                 TIME_STEP      = 100,
                 CELL_UNITS     = [150, 200],
                 FULL_UNITS     = [500, 600],
                 KEEP_PROB      = 0.5,
                 OUTPUT_NUM     = 1, ):

        # Definition Params:
        self.MODEL_TYPE   = MODEL_TYPE      # Classification or Regression (str)
        self.FILTER_NUM   = FILTER_NUM      # CNN Filter Number
        self.FILTER_SIZE  = FILTER_SIZE     # CNN Filter Height Size
        self.POOL_SIZE    = POOL_SIZE       # CNN Pool Height size
        self.INPUT_LENGTH = INPUT_LENGTH    # Sensor Number * Feature Number
        self.TIME_STEP    = TIME_STEP       # Window Number
        self.CELL_UNITS   = CELL_UNITS      # LSTM Cell Units (List)
        self.FULL_UNITS   = FULL_UNITS      # FC Units (List)
        self.KEEP_PROB    = KEEP_PROB       # Drop Layer Keep Probability
        self.OUTPUT_NUM   = OUTPUT_NUM      # Net Output Size

        # Define Net Input:
        self.X = tf.placeholder(shape=[None, self.TIME_STEP, self.INPUT_LENGTH, 1], dtype=tf.float32, name='input_x')
        if self.MODEL_TYPE == 'Regression':
            self.y = tf.placeholder(shape=[None, ], dtype=tf.float32, name='true_y')
        else:
            self.y = tf.placeholder(shape=[None, self.OUTPUT_NUM], dtype=tf.float32, name='true_y')
        self.keep_pro = tf.placeholder(shape=None, dtype=tf.float32, name='keep_pro')

        # Construct Net:
        with tf.variable_scope('cnn-layer', reuse=tf.AUTO_REUSE):
            self.cnn_output = self.cnn_layer(cnn_input = self.X,
                                             k         = self.FILTER_NUM,
                                             m         = self.FILTER_SIZE,
                                             s         = self.POOL_SIZE,
                                             d         = self.INPUT_LENGTH, )
        with tf.variable_scope('bilstm-layer', reuse=tf.AUTO_REUSE):
            self.bilstm_output = self.bilstm_layer(bilstm_input = self.cnn_output,
                                                   num_units    = self.CELL_UNITS, )
        with tf.variable_scope('fc-layer', reuse=tf.AUTO_REUSE):
            self.fc_output = self.fc_layer(fc_input  = self.bilstm_output,
                                           num_units = self.FULL_UNITS,
                                           keep_prob = self.keep_pro, )
        with tf.variable_scope('cost-function', reuse=tf.AUTO_REUSE):
            self.create_cost_function()

    def create_cost_function(self):
        # Calculate Cost function:
        fc_out = tf.layers.dense(self.fc_output, self.OUTPUT_NUM,
                                 activation=None, use_bias=True,
                                 kernel_initializer=tf.glorot_uniform_initializer())
        if self.MODEL_TYPE == 'Regression':
            self.pred = tf.reshape(fc_out, shape=[-1, ])
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pred)))
            # self.cost = tf.reduce_mean(tf.square(self.y - self.pred))
        else:
            self.softmax_y = tf.nn.softmax(fc_out, axis=1)
            self.pred = tf.argmax(self.softmax_y, axis=1)
            self.cost = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(self.y, tf.log(self.softmax_y + 1e-8)),1))
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(self.cost)

    def train_model(self,
                    train_x     = None,      # shape: [-1, num_steps, data_length, 1]
                    train_y     = None,
                    test_x      = None,      # shape: [-1, num_steps, data_length, 1]
                    test_y      = None,
                    batch_size  = None,
                    num_epochs  = None,      # the total training steps = (n_samples/batch_size)*num_epochs
                    num_threads = None, ):   # don't support -1
        x_batch, y_batch = self.get_Batch(data        = train_x,
                                          label       = train_y,
                                          batch_size  = batch_size,
                                          num_epochs  = num_epochs,
                                          num_threads = num_threads, )
        sess = self.model_init()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        epoch = 0
        # last_cost = 1000
        # early_epoch = 0
        try:
            while not coord.should_stop():
                data, label = sess.run([x_batch, y_batch])
                sess.run(self.train_op, feed_dict={self.X: data, self.y: label, self.keep_pro: self.KEEP_PROB})
                train_pre, train_cost = sess.run([self.pred, self.cost],
                                                 feed_dict={self.X: data,
                                                            self.y: label,
                                                            self.keep_pro: 1.0})
                test_pre, test_cost = sess.run([self.pred, self.cost],
                                               feed_dict={self.X: test_x,
                                                          self.y: test_y,
                                                          self.keep_pro: 1.0})
                epoch = epoch + 1
                if epoch % 50 == 0:
                    if self.MODEL_TYPE == 'Regression':
                        print("Epoch %d, Training mse %g, Testing mse %g" % (epoch, train_cost, test_cost))
                    else:
                        print("Epoch %d, Training cost %g, Testing cost %g" % (epoch, train_cost, test_cost))
                # if np.abs(test_cost - last_cost) < 0.01:
                #     early_epoch += 1
                #     if early_epoch > 100:
                #         break
                # else:
                #     early_epoch = 0
                # last_cost = test_cost

        except tf.errors.OutOfRangeError:
            print("---Train end---")
        finally:
            coord.request_stop()
            print('---Program end---')
        coord.join(threads)

    @staticmethod
    def model_init():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess
    @staticmethod
    def get_Batch(data=None, label=None, batch_size=None, num_epochs=None, num_threads=None):
        input_queue = tf.train.slice_input_producer([data, label], num_epochs=num_epochs,
                                                    shuffle=True, capacity=32)
        x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size,
                                          num_threads=num_threads, capacity=32,
                                          allow_smaller_final_batch=False)
        return x_batch, y_batch
    @staticmethod
    def cnn_layer(cnn_input=None, k=None, m=None, s=None, d=None):
        cnn1 = tf.contrib.layers.conv2d(cnn_input,
                                        num_outputs=k,
                                        kernel_size=[m, d],
                                        stride=[1, d],
                                        padding='VALID', )

        cnn1_pool = tf.nn.max_pool(cnn1,
                                   ksize=[1, s, 1, 1],
                                   strides=[1, s, 1, 1],
                                   padding='SAME',
                                   name='cnn1_max_pool')

        cnn1_shape = cnn1_pool.get_shape()
        cnn_out = tf.reshape(cnn1_pool, shape=[-1, cnn1_shape[1], cnn1_shape[-1]])
        return cnn_out
    @staticmethod
    def bilstm_layer(bilstm_input=None, num_units=None):
        # first bi-lstm cell
        with tf.variable_scope('1st-bi-lstm-layer', reuse=tf.AUTO_REUSE):
            cell_fw_1 = tf.nn.rnn_cell.LSTMCell(num_units=num_units[0], state_is_tuple=True)
            cell_bw_1 = tf.nn.rnn_cell.LSTMCell(num_units=num_units[0], state_is_tuple=True)
            outputs_1, states_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw_1, cell_bw_1, inputs=bilstm_input,
                                                                  dtype=tf.float32)

        # second bi-lstm cell
        with tf.variable_scope('2nd-bi-lstm-layer', reuse=tf.AUTO_REUSE):
            # input_2 = tf.add(outputs_1[0], outputs_1[1])
            input_2 = tf.concat([outputs_1[0], outputs_1[1]], axis=2)
            cell_fw_2 = tf.nn.rnn_cell.LSTMCell(num_units=num_units[1], state_is_tuple=True)
            cell_bw_2 = tf.nn.rnn_cell.LSTMCell(num_units=num_units[1], state_is_tuple=True)
            outputs_2, states_2 = tf.nn.bidirectional_dynamic_rnn(cell_fw_2, cell_bw_2, inputs=input_2,
                                                                  dtype=tf.float32)

        # bilstm output
        with tf.variable_scope('bi-lstm-layer-output', reuse=tf.AUTO_REUSE):
            bilstm_out = tf.concat([states_2[0].h, states_2[1].h], axis=1)
        return bilstm_out
    @staticmethod
    def fc_layer(fc_input=None, num_units=None, keep_prob=None):
        fc_input_ = tf.nn.dropout(fc_input, keep_prob=keep_prob)
        fc1 = tf.layers.dense(fc_input_, num_units[0], activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())
        fc1_ = tf.nn.dropout(fc1, keep_prob=keep_prob)
        fc_out = tf.layers.dense(fc1_, num_units[1], activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_uniform_initializer())
        # fc_out = tf.layers.dense(fc_out, 1, activation=None, use_bias=False,
        #                          kernel_initializer=tf.glorot_normal_initializer())
        return fc_out


