# -*- coding: utf-8

import numpy as np
import pickle
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

class FBGRU(object):
    def __init__(self,
                 MODEL_TYPE     = 'Regression',
                 INPUT_LENGTH   = 70,
                 TIME_STEP      = 20,
                 CELL_UNITS     = 100,
                 FULL_UNITS     = [100, 400],
                 KEEP_PROB      = 0.5,
                 OUTPUT_NUM     = 1, ):

        # Definition Params:
        self.MODEL_TYPE   = MODEL_TYPE      # Classification or Regression (str)
        self.INPUT_LENGTH = INPUT_LENGTH    # Sensor Number * Feature Number
        self.TIME_STEP    = TIME_STEP       # Window Number
        self.CELL_UNITS   = CELL_UNITS      # GRU Cell Units
        self.FULL_UNITS   = FULL_UNITS      # FC Units (List)
        self.KEEP_PROB    = KEEP_PROB       # Drop Layer Keep Probability
        self.OUTPUT_NUM   = OUTPUT_NUM      # Net Output Size

        # Define Net Input:
        self.X = tf.placeholder(shape=[None, self.TIME_STEP, self.INPUT_LENGTH], dtype=tf.float32, name='input_x')
        self.X_ = tf.placeholder(shape=[None, self.INPUT_LENGTH], dtype=tf.float32, name='input_x_')
        if self.MODEL_TYPE == 'Regression':
            self.y = tf.placeholder(shape=[None, ], dtype=tf.float32, name='true_y')
        else:
            self.y = tf.placeholder(shape=[None, self.OUTPUT_NUM], dtype=tf.float32, name='true_y')
        self.keep_pro = tf.placeholder(shape=None, dtype=tf.float32, name='keep_pro')

        # Construct Net:
        with tf.variable_scope('bigru-layer', reuse=tf.AUTO_REUSE):
            self.bigru_output = self.bigru_layer(bilstm_input = self.X,
                                                  num_units    = self.CELL_UNITS, )
        with tf.variable_scope('fc1-layer', reuse=tf.AUTO_REUSE):
            # self.weight_average_out = self.get_weight_average_layer(weight_average_input=self.X)
            self.fc1_output = self.fc_layer_1(fc_input  = self.X_,
                                              num_units = self.FULL_UNITS[0],
                                              keep_prob = self.keep_pro, )
        with tf.variable_scope('fc2-layer', reuse=tf.AUTO_REUSE):
            self.fc_layer_input = tf.concat([self.bigru_output, self.fc1_output], axis=1)
            self.fc2_output = self.fc_layer_2(fc_input  = self.fc_layer_input,
                                              num_units = self.FULL_UNITS[1],
                                              keep_prob = self.keep_pro, )
        with tf.variable_scope('cost-function', reuse=tf.AUTO_REUSE):
            self.create_cost_function()

    def create_cost_function(self):
        # Calculate Cost function:
        fc_out = tf.layers.dense(self.fc2_output, self.OUTPUT_NUM,
                                 activation=None, use_bias=True,
                                 kernel_initializer=tf.glorot_uniform_initializer())
        if self.MODEL_TYPE == 'Regression':
            self.pred = tf.reshape(fc_out, shape=[-1, ])
            # self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pred)))
            self.cost = tf.reduce_mean(tf.square(self.y - self.pred))
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
        last_cost = 1000
        early_epoch = 0
        try:
            while not coord.should_stop():
                data, label = sess.run([x_batch, y_batch])
                sess.run(self.train_op, feed_dict={self.X: data,
                                                   self.y: label,
                                                   self.X_: self.get_weight_average_layer(data),
                                                   self.keep_pro: self.KEEP_PROB})
                train_pre, train_cost = sess.run([self.pred, self.cost],
                                                 feed_dict={self.X: data,
                                                            self.X_: self.get_weight_average_layer(data),
                                                            self.y: label,
                                                            self.keep_pro: 1.0})
                test_pre, test_cost = sess.run([self.pred, self.cost],
                                               feed_dict={self.X: test_x,
                                                          self.X_:self.get_weight_average_layer(test_x),
                                                          self.y: test_y,
                                                          self.keep_pro: 1.0})
                epoch = epoch + 1
                if epoch % 50 == 0:
                    if self.MODEL_TYPE == 'Regression':
                        print("Epoch %d, Training mse %g, Testing mse %g" % (epoch, train_cost, test_cost))
                    else:
                        print("Epoch %d, Training cost %g, Testing cost %g" % (epoch, train_cost, test_cost))
                if np.abs(test_cost - last_cost) < 0.01:
                    early_epoch += 1
                    if early_epoch > 100:
                        break
                else:
                    early_epoch = 0
                last_cost = test_cost

        except tf.errors.OutOfRangeError:
            print("---Train end---")
        finally:
            coord.request_stop()
            print('---Program end---')
        coord.join(threads)

    @staticmethod
    def model_init():
        # graph = tf.Graph()
        # tf.set_random_seed(seed=1)
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
    def bigru_layer(bilstm_input=None, num_units=None):
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units=num_units, name='fw')
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units=num_units, name='bw')
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=bilstm_input, dtype=tf.float32)
        bigru_out = tf.concat([states[0], states[1]], axis=1)
        return bigru_out
    @staticmethod
    def get_weight_average_layer(weight_average_input=None):
        _arr_weight_average_input = np.array(weight_average_input)
        _, T, _ = _arr_weight_average_input.shape
        _arr = []
        for ck in _arr_weight_average_input:    # every batch
            qk = np.array([np.exp(np.min([k - 1, T - k])) for k in range(1, T+1)])
            sigma_qk = np.sum(qk, dtype=np.float32)
            wk = np.array([qj * 1.0 / sigma_qk for qj in qk])
            c = np.array([wk[k]*ck[k] for k in range(T)]).sum(axis=0)
            _arr.append(c)
        return np.array(_arr)
    @staticmethod
    def fc_layer_1(fc_input=None, num_units=None, keep_prob=None):
        fc_input_ = tf.nn.dropout(fc_input, keep_prob=keep_prob)
        fc = tf.layers.dense(fc_input_, num_units, activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer())
        fc_out = tf.nn.dropout(fc, keep_prob=keep_prob)
        return fc_out
    @staticmethod
    def fc_layer_2(fc_input=None, num_units=None, keep_prob=None):
        fc_input_ = tf.nn.dropout(fc_input, keep_prob=keep_prob)
        fc_out = tf.layers.dense(fc_input_, num_units, activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_uniform_initializer())
        # fc_out = tf.nn.dropout(fc, keep_prob=keep_prob)
        return fc_out


