from collections import deque
import os

import numpy as np
import tensorflow.compat.v1 as tf

class DRLactor:
    def __init__(self,n_features, n_actions, lr, num, lbd, rho):
        self.global_ = tf.Variable(tf.constant(0))
        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.num = str(num)
        self.lbd = str(lbd)
        self.rho = str(rho)
        self.init_model(n_actions, lr)
        self.enable_actions = list(range(n_actions))
        self.action_size = n_actions

    def init_model(self, n_actions, lr):

        s_norm_h = tf.nn.l2_normalize(self.s, dim=0)
        s_norm = tf.nn.l2_normalize(s_norm_h, dim=1)


        self.l1 = tf.layers.dense(
            inputs=s_norm,
            units=200,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
            # name='l1'
        )

        self.l2 = tf.layers.dense(
            inputs=s_norm,
            units=100,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
            # name='l1'
        )

        self.acts_prob = tf.layers.dense(
            inputs=self.l2,
            units=n_actions,
            activation=tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),

        )


        # self.log_prob = tf.log(self.acts_prob[0, self.a])

        self.log_prob = tf.log(tf.clip_by_value(self.acts_prob[0, self.a], 1e-8, 1.0))


        self.exp_v = tf.reduce_mean(self.log_prob * self.td_error)


        self.lr_dc = tf.train.exponential_decay(lr, self.global_, 50000, 0.7, staircase=True)
        self.train_op = tf.train.AdamOptimizer(self.lr_dc).minimize(-self.exp_v)


        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())

        self.modelDir = "./model_AC_cost_" + self.num + '_' + self.lbd + '_' + self.rho + "/actor.ckpt"
        self.saver = tf.train.Saver()

    def learn(self, s, a, td, i):
        feed_dict = {self.s: s, self.a: a, self.td_error: td, self.global_: i}
        _, exp_v, _, _ = self.sess.run([self.train_op, self.exp_v, self.acts_prob, self.lr_dc], feed_dict)

        return exp_v

    def choose_action(self, s, delta_epsilon):
        if np.random.rand() <= delta_epsilon:  # random
            action = np.random.choice(self.enable_actions)
        
        else:
            probs, _ = self.sess.run([self.acts_prob, self.l1], {self.s: s})
            action = np.random.choice(self.enable_actions,p=probs[0])

        return action

    # def choose_action(self, s):
    #     probs, _ = self.sess.run([self.acts_prob, self.l1], {self.s: s})
    #     action = np.random.choice(self.action_size, 1, p=probs[0]).take(0)
    #     print('check probs', probs)
    #     #
    #     # if np.random.rand() <= 0.3:  # random
    #     #     action_list = list(range(self.action_size))
    #     #
    #     #     action = np.random.choice(action_list)
    #     #
    #     # else:
    #     #     action = np.argmax(probs)
    #
    #
    #     return action

    def save_model(self):
        self.save_path = self.saver.save(self.sess, self.modelDir)
        print('actor saved')

    def reload_model(self):
        self.saver.restore(self.sess, self.modelDir)
        print('actor reloaded')

    def kill_graph(self):
        tf.reset_default_graph()
