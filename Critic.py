from collections import deque
import os

import numpy as np
import tensorflow.compat.v1 as tf

class DRLcritic:
    def __init__(self, n_features, lr, num, lbd, rho):
        self.global_ = tf.Variable(tf.constant(0))
        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.GAMMA = 0.9
        self.num = str(num)
        self.lbd = str(lbd)
        self.rho = str(rho)
        self.init_model(lr)

    def init_model(self, lr):
        with tf.variable_scope('Critic'):
            s_norm_h = tf.nn.l2_normalize(self.s, dim=0)
            s_norm = tf.nn.l2_normalize(s_norm_h, dim=1)



            l1 = tf.layers.dense(
                inputs=s_norm,
                units=200,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                # name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=100,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                # name='l2'
            )

            self.v = tf.layers.dense(
                inputs=l2,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                # name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.lr_dc = tf.train.exponential_decay(lr, self.global_, 50000, 0.7, staircase=True)
            self.train_op = tf.train.AdamOptimizer(self.lr_dc).minimize(self.loss)


        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())

        self.modelDir = "./model_AC_cost_" + self.num + '_' + self.lbd + '_' + self.rho + "/critic.ckpt"
        self.saver = tf.train.Saver()

    def learn(self, s, r, s_, i):
        # print('check critic ', s)
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _, _ = self.sess.run([self.td_error, self.train_op, self.lr_dc],
                                        {self.s: s, self.v_: v_, self.r: r, self.global_: i})

        return td_error

    def save_model(self):
        self.save_path = self.saver.save(self.sess, self.modelDir)
        print('critic saved')

    def reload_model(self):
        self.saver.restore(self.sess, self.modelDir)
        print('critic reloaded')

    def kill_graph(self):
        tf.reset_default_graph()