import mher.common.tf_util as U
import tensorflow as tf
import numpy as np
from mher.common import logger
from mher.common.mpi_adam import MpiAdam
from mher.algo.util import store_args
from mher.algo.normalizer import NormalizerNumpy
from tensorflow.keras.layers import LSTM


def nnlstm(input, layers_sizes, n, reuse=None, flatten=False, use_layer_norm=False, name=""):
    input = tf.reshape(input, [-1, 16, input.shape[1]])

    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        norm = tf.contrib.layers.layer_norm if i < len(layers_sizes) - 1 else None
        with tf.variable_scope(name + '_lstm_' + str(i)):
            lstm = LSTM(size,
                   activation=activation,
                   return_sequences=True,
                   recurrent_dropout=0.1,
                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                   recurrent_regularizer=tf.keras.regularizers.l2(0.01),
                   bias_regularizer=tf.keras.regularizers.l2(0.01))
            input = lstm(input)

        if use_layer_norm and norm:
            input = norm(input, reuse=reuse, scope=name + '_layer_norm_' + str(i))
        if activation:
            input = activation(input)

    input = tf.layers.dense(inputs=input,
                            units=n*size,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),

                            reuse=reuse,
                            name=name + '_' + str(i))
    if use_layer_norm and norm:
        input = norm(input, reuse=reuse, scope=name + '_layer_norm_' + str(i))
    if activation:
        input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    input = tf.reshape(input, [-1, n*size])

    return input

def lstm(input, layers_sizes, reuse=None, flatten=False, use_layer_norm=False, name=""):
    input = tf.reshape(input, [-1, 16, input.shape[1]])

    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        norm = tf.contrib.layers.layer_norm if i < len(layers_sizes) - 1 else None
        with tf.variable_scope(name + '_lstm_' + str(i)):
            lstm = LSTM(size,
                   activation=activation,
                   return_sequences=True,
                   recurrent_dropout=0.1,
                   kernel_regularizer=tf.keras.regularizers.l2(0.01),
                   recurrent_regularizer=tf.keras.regularizers.l2(0.01),
                   bias_regularizer=tf.keras.regularizers.l2(0.01))
            input = lstm(input)

        if use_layer_norm and norm:
            input = norm(input, reuse=reuse, scope=name + '_layer_norm_' + str(i))
        if activation:
            input = activation(input)

    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    input = tf.reshape(input, [-1, size])

    return input

def nn(input, layers_sizes, reuse=None, flatten=False, use_layer_norm=False, name=""):
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        norm = tf.contrib.layers.layer_norm if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),

                                reuse=reuse,
                                name=name + '_' + str(i))
        if use_layer_norm and norm:
            input = norm(input, reuse=reuse, scope=name + '_layer_norm_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def _vars(scope):
    res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(res) > 0
    return res



# numpy forward dynamics
class ForwardDynamicsNumpy:
    @store_args
    def __init__(self, dimo, dimu, method, n, clip_norm=5, norm_eps=1e-4, hidden=256, layers=4, learning_rate=1e-3, name='1'):
        self.obs_normalizer = NormalizerNumpy(size=dimo, eps=norm_eps)
        self.action_normalizer = NormalizerNumpy(size=dimu, eps=norm_eps)
        self.sess = U.get_session()
        self.name = name

        with tf.variable_scope('forward_dynamics_numpy_' + self.name):
            self.obs_norm = []
            for i in range(n+1):
                obs_placeholder = tf.placeholder(tf.float32, shape=(None, self.dimo), name=f'obs{i}_norm')
                self.obs_norm.append(obs_placeholder)
            self.actions_norm = tf.placeholder(tf.float32, shape=(None,self.dimu) , name='actions')

            self.dynamics_scope = tf.get_variable_scope().name
            input = tf.concat(values=[self.obs_norm[0], self.actions_norm], axis=-1)
            if (method=='lstm'):
                self.next_state_diff_tf = lstm(input, [hidden] * layers + [self.dimo])
                self.next_state_norm_tf = self.next_state_diff_tf + self.obs_norm[0]
                # loss functions
                self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf - self.obs_norm[1] + self.obs_norm[0]), axis=1)
            elif (method=='multi'):
                self.next_state_diff_tf = nnlstm(input, [hidden] * layers + [self.dimo], n)
                self.next_state_norm_tfs = []
                for i in range(n):
                    next_state_norm_tfi = self.next_state_diff_tf[:,int(i*self.next_state_diff_tf.shape[1].value / n):int((i+1) * (self.next_state_diff_tf.shape[1].value / n))] + self.obs_norm[i]
                    self.next_state_norm_tfs.append(next_state_norm_tfi)
                self.next_state_norm_tf = sum(self.next_state_norm_tfs)/n

                # loss functions
                self.next_state_loss_tfs = []
                for i in range(n):
                    next_state_loss_tfi = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:,int(i*(self.next_state_diff_tf.shape[1].value / n)):int((i+1) * (self.next_state_diff_tf.shape[1].value / n))] - self.obs_norm[i+1]+ self.obs_norm[i]), axis=1)
                    self.next_state_loss_tfs.append(next_state_loss_tfi)
                self.per_sample_loss_tf = sum(self.next_state_loss_tfs) / n
            elif (method=='avg'):
                self.next_state_diff_tf = nnlstm(input, [hidden] * layers + [self.dimo], n)
                self.next_state_norm_tfs = []
                for i in range(n):
                    next_state_norm_tfi = self.next_state_diff_tf[:,int(i*self.next_state_diff_tf.shape[1].value / n):int((i+1) * (self.next_state_diff_tf.shape[1].value / n))] + self.obs_norm[0]
                    self.next_state_norm_tfs.append(next_state_norm_tfi)
                self.next_state_norm_tf = sum(self.next_state_norm_tfs)/n

                # loss functions
                self.next_state_loss_tfs = []
                for i in range(n):
                    next_state_loss_tfi = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:,int(i*(self.next_state_diff_tf.shape[1].value / n)):int((i+1) * (self.next_state_diff_tf.shape[1].value / n))] - self.obs_norm[1]+ self.obs_norm[0]), axis=1)
                    self.next_state_loss_tfs.append(next_state_loss_tfi)
                self.per_sample_loss_tf = sum(self.next_state_loss_tfs) / n
            elif (method == 'default'):
                self.next_state_diff_tf = nn(input, [hidden] * layers + [self.dimo])
                self.next_state_norm_tf = self.next_state_diff_tf + self.obs_norm[0]
                # loss functions
                self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf - self.obs_norm[1] + self.obs_norm[0]), axis=1)

        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)
        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.dynamics_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.dynamics_scope), scale_grad_by_procs=False)
        # initial
        tf.variables_initializer(_vars(self.dynamics_scope)).run()
        self.dynamics_adam.sync()

    def predict_next_state(self, obs0, actions, transitions, n):
        obs0_norm = self.obs_normalizer.normalize(obs0)
        obs_norm = []
        obs_norm.append(obs0_norm)
        for i in range(2, (n + 2)):
            obsi_norm = self.obs_normalizer.normalize(transitions[f'o_{i}'])
            obs_norm.append(obsi_norm)
        self.action_normalizer.update(actions)
        action_norm = self.action_normalizer.normalize(actions)
        feed_dict = {self.actions_norm: action_norm}

        for i in range(n + 1):
            feed_dict[self.obs_norm[i]] = obs_norm[i]
        obs1 = self.sess.run(self.next_state_norm_tf, feed_dict=feed_dict)
        obs1_norm = self.obs_normalizer.denormalize(obs1)
        return obs1_norm

    def clip_gauss_noise(self, size):
        return 0

    def update(self, obs0, actions, transitions, n, times=1):
        self.obs_normalizer.update(obs0)
        for i in range(2, (n+2)):
            self.obs_normalizer.update(transitions[f'o_{i}'])
        self.action_normalizer.update(actions)

        for _ in range(times):
             # use small noise for smooth
            obs0_norm = self.obs_normalizer.normalize(obs0) + self.clip_gauss_noise(size=self.dimo)
            action_norm = self.action_normalizer.normalize(actions) + self.clip_gauss_noise(size=self.dimu)
            obs_norm = []
            obs_norm.append(obs0_norm)
            for i in range(2,(n+2)):
                 obsi_norm = self.obs_normalizer.normalize(transitions[f'o_{i}'])
                 obs_norm.append(obsi_norm)
            feed_dict = {self.actions_norm: action_norm}

            for i in range(n+1):
                 feed_dict[self.obs_norm[i]] = obs_norm[i]

            dynamics_grads, dynamics_loss, dynamics_per_sample_loss = self.sess.run(
                    [self.dynamics_grads, self.mean_loss_tf, self.per_sample_loss_tf],
                    feed_dict=feed_dict)
            self.dynamics_adam.update(dynamics_grads, stepsize=self.learning_rate)
        return dynamics_loss


class EnsembleForwardDynamics:
    @store_args
    def __init__(self, num_models, dimo, dimu, method, n, clip_norm=5, norm_eps=1e-4, hidden=256, layers=4, learning_rate=1e-3):
        self.num_models = num_models
        self.models = []
        for i in range(num_models):
            self.models.append(ForwardDynamicsNumpy(dimo, dimu, method, n, clip_norm, norm_eps, hidden, layers, learning_rate, name=str(i)))

    def predict_next_state(self, obs0, actions, transitions, n, mode='mean'):
        # random select prediciton or mean prediction
        if mode == 'random':
            idx = int(np.random.random() * self.num_models)
            model = self.models[idx]
            result = model.predict_next_state(obs0, actions, transitions, n)
        elif mode == 'mean':
            res = []
            for model in self.models:
                res.append(model.predict_next_state(obs0, actions, transitions, n))

            result_array = np.array(res).transpose([1,0,2])
            result = result_array.mean(axis=1)
        elif mode == 'mean_std':
            res = []
            for model in self.models:
                res.append(model.predict_next_state(obs0, actions, transitions, n))
            result_array = np.array(res).transpose([1,0,2])
            result = result_array.mean(axis=1)
            std = result_array.std(axis=1).sum(axis=1)
            return result, std
        else:
            raise NotImplementedError('No such prediction mode!')
        return result


    def update(self, obs0, actions,transitions, n, times=1, mode='random'):
        # update all or update a random model
        if mode == 'all':
            dynamics_per_sample_loss = []
            for model in self.models:
                loss = model.update(obs0, actions, transitions, n, times)
                dynamics_per_sample_loss.append(loss)
            dynamics_per_sample_loss_array = np.array(dynamics_per_sample_loss)
            dynamics_per_sample_loss = dynamics_per_sample_loss_array.mean(axis=0)
        elif mode == 'random':
            idx = int(np.random.random() * self.num_models)
            model = self.models[idx]
            dynamics_per_sample_loss = model.update(obs0, actions, transitions, n,times)
        else:
            raise NotImplementedError('No such update mode!')
        return dynamics_per_sample_loss
