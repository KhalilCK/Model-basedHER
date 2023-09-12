import mher.common.tf_util as U
import tensorflow as tf
import numpy as np
from mher.common import logger
from mher.common.mpi_adam import MpiAdam
from mher.algo.util import store_args
from mher.algo.normalizer import NormalizerNumpy
from tensorflow.keras.layers import LSTM


def nnlstm(input, layers_sizes, reuse=None, flatten=False, use_layer_norm=False, name=""):
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
                            units=5*size,
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
    input = tf.reshape(input, [-1, 5*size])

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
    def __init__(self, dimo, dimu, t, clip_norm=5, norm_eps=1e-4, hidden=256, layers=4, learning_rate=1e-3, name='1'):
        self.obs_normalizer = NormalizerNumpy(size=dimo, eps=norm_eps)
        self.action_normalizer = NormalizerNumpy(size=dimu, eps=norm_eps)
        self.sess = U.get_session()
        self.name = name

        with tf.variable_scope('forward_dynamics_numpy_' + self.name):
            self.obs0_norm = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs0')
            self.obs1_norm = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs1')
            self.obs2_norm = tf.placeholder(tf.float32, shape=(None, self.dimo), name='obs2')
            self.obs3_norm = tf.placeholder(tf.float32, shape=(None, self.dimo), name='obs3')
            self.obs4_norm = tf.placeholder(tf.float32, shape=(None, self.dimo), name='obs4')
            self.obs5_norm = tf.placeholder(tf.float32, shape=(None, self.dimo), name='obs5')
            self.actions_norm = tf.placeholder(tf.float32, shape=(None,self.dimu) , name='actions')

            self.dynamics_scope = tf.get_variable_scope().name
            input = tf.concat(values=[self.obs0_norm, self.actions_norm], axis=-1)
            if (t==1):
                self.next_state_diff_tf = lstm(input, [hidden] * layers + [self.dimo])
                self.next_state_norm_tf = self.next_state_diff_tf + self.obs0_norm
                # loss functions
                self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf - self.obs1_norm + self.obs0_norm), axis=1)
            elif (t==2):
                self.next_state_diff_tf = nnlstm(input, [hidden] * layers + [self.dimo])

                self.next_state_norm_tf1 = self.next_state_diff_tf[:,0:int((1 / 5) * (self.next_state_diff_tf.shape[1].value))] + self.obs0_norm
                self.next_state_norm_tf2 = self.next_state_diff_tf[:,int(self.next_state_diff_tf.shape[1].value / 5):int(2 * (self.next_state_diff_tf.shape[1].value / 5))] + self.obs0_norm
                self.next_state_norm_tf3 = self.next_state_diff_tf[:,int(2 * (self.next_state_diff_tf.shape[1].value / 5)):int(3 * (self.next_state_diff_tf.shape[1].value / 5))] + self.obs0_norm
                self.next_state_norm_tf4 = self.next_state_diff_tf[:,int(3 * (self.next_state_diff_tf.shape[1].value / 5)):int(4 * (self.next_state_diff_tf.shape[1].value / 5))] + self.obs0_norm
                self.next_state_norm_tf5 = self.next_state_diff_tf[:,int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(self.next_state_diff_tf.shape[1].value)] + self.obs0_norm
                self.next_state_norm_tf = (self.next_state_norm_tf1 + self.next_state_norm_tf2 + self.next_state_norm_tf3 + self.next_state_norm_tf4 + self.next_state_norm_tf5) / 5
                # loss functions
                self.per_sample_loss_tf1 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, 0:int(1 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs1_norm + self.obs0_norm), axis=1)
                self.per_sample_loss_tf2 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:,int((self.next_state_diff_tf.shape[1].value / 5)):int(2 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs2_norm + self.obs1_norm), axis=1)
                self.per_sample_loss_tf3 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(2 * (self.next_state_diff_tf.shape[1].value / 5)):int(3 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs3_norm + self.obs2_norm), axis=1)
                self.per_sample_loss_tf4 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(3 * (self.next_state_diff_tf.shape[1].value / 5)):int(4 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs4_norm + self.obs3_norm), axis=1)
                self.per_sample_loss_tf5 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(5 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs5_norm + self.obs4_norm), axis=1)
                self.per_sample_loss_tf = (self.per_sample_loss_tf1 + self.per_sample_loss_tf2 + self.per_sample_loss_tf3 + self.per_sample_loss_tf4 + self.per_sample_loss_tf5) / 5
            elif (t==3):
                self.next_state_diff_tf = nnlstm(input, [hidden] * layers + [self.dimo])

                self.next_state_norm_tf1 = self.next_state_diff_tf[:,0:int((1 / 5) * (self.next_state_diff_tf.shape[1].value))] + self.obs0_norm
                self.next_state_norm_tf2 = self.next_state_diff_tf[:,int(self.next_state_diff_tf.shape[1].value / 5):int(2 * (self.next_state_diff_tf.shape[1].value / 5))] + self.obs0_norm
                self.next_state_norm_tf3 = self.next_state_diff_tf[:,int(2 * (self.next_state_diff_tf.shape[1].value / 5)):int(3 * (self.next_state_diff_tf.shape[1].value / 5))] + self.obs0_norm
                self.next_state_norm_tf4 = self.next_state_diff_tf[:,int(3 * (self.next_state_diff_tf.shape[1].value / 5)):int(4 * (self.next_state_diff_tf.shape[1].value / 5))] + self.obs0_norm
                self.next_state_norm_tf5 = self.next_state_diff_tf[:,int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(self.next_state_diff_tf.shape[1].value)] + self.obs0_norm
                self.next_state_norm_tf = (self.next_state_norm_tf1 + self.next_state_norm_tf2 + self.next_state_norm_tf3 + self.next_state_norm_tf4 + self.next_state_norm_tf5) / 5
                # loss functions
                self.per_sample_loss_tf1 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, 0:int(1 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs1_norm + self.obs0_norm), axis=1)
                self.per_sample_loss_tf2 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:,int((self.next_state_diff_tf.shape[1].value / 5)):int(2 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs2_norm + self.obs0_norm), axis=1)
                self.per_sample_loss_tf3 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(2 * (self.next_state_diff_tf.shape[1].value / 5)):int(3 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs3_norm + self.obs0_norm), axis=1)
                self.per_sample_loss_tf4 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(3 * (self.next_state_diff_tf.shape[1].value / 5)):int(4 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs4_norm + self.obs0_norm), axis=1)
                self.per_sample_loss_tf5 = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(5 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs5_norm + self.obs0_norm), axis=1)
                self.per_sample_loss_tf = (self.per_sample_loss_tf1 + self.per_sample_loss_tf2 + self.per_sample_loss_tf3 + self.per_sample_loss_tf4 + self.per_sample_loss_tf5) / 5
            elif (t==4):
                self.next_state_diff_tf = nnlstm(input, [hidden] * layers + [self.dimo])
                self.next_state_norm_tf = self.next_state_diff_tf[:,int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(self.next_state_diff_tf.shape[1].value)] + self.obs0_norm
                self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(5 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs5_norm + self.obs4_norm), axis=1)
            elif (t==5):
                self.next_state_diff_tf = nnlstm(input, [hidden] * layers + [self.dimo])
                self.next_state_norm_tf = self.next_state_diff_tf[:,int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(self.next_state_diff_tf.shape[1].value)] + self.obs0_norm
                self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf[:, int(4 * (self.next_state_diff_tf.shape[1].value / 5)):int(5 * (self.next_state_diff_tf.shape[1].value / 5))] - self.obs5_norm + self.obs0_norm), axis=1)
            else:
                self.next_state_diff_tf = nn(input, [hidden] * layers + [self.dimo])
                self.next_state_norm_tf = self.next_state_diff_tf + self.obs0_norm
                # loss functions
                self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf - self.obs1_norm + self.obs0_norm), axis=1)

        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)
        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.dynamics_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.dynamics_scope), scale_grad_by_procs=False)
        # initial
        tf.variables_initializer(_vars(self.dynamics_scope)).run()
        self.dynamics_adam.sync()

    def predict_next_state(self, obs0, actions):
        obs0_norm = self.obs_normalizer.normalize(obs0)
        action_norm = self.action_normalizer.normalize(actions)
        obs1 = self.sess.run(self.next_state_norm_tf, feed_dict={
            self.obs0_norm: obs0_norm,
            self.actions_norm:action_norm
        })
        obs1_norm = self.obs_normalizer.denormalize(obs1)
        return obs1_norm

    def clip_gauss_noise(self, size):
        return 0

    def update(self, obs0, actions, obs1, obs2, obs3, obs4, obs5, times=1):
        self.obs_normalizer.update(obs0)
        self.obs_normalizer.update(obs1)
        self.obs_normalizer.update(obs2)
        self.obs_normalizer.update(obs3)
        self.obs_normalizer.update(obs4)
        self.obs_normalizer.update(obs5)
        self.action_normalizer.update(actions)

        for _ in range(times):
             # use small noise for smooth
            obs0_norm = self.obs_normalizer.normalize(obs0) + self.clip_gauss_noise(size=self.dimo)
            action_norm = self.action_normalizer.normalize(actions) + self.clip_gauss_noise(size=self.dimu)
            obs1_norm = self.obs_normalizer.normalize(obs1)
            obs2_norm = self.obs_normalizer.normalize(obs2)
            obs3_norm = self.obs_normalizer.normalize(obs3)
            obs4_norm = self.obs_normalizer.normalize(obs4)
            obs5_norm = self.obs_normalizer.normalize(obs5)

            dynamics_grads, dynamics_loss, dynamics_per_sample_loss = self.sess.run(
                    [self.dynamics_grads, self.mean_loss_tf, self.per_sample_loss_tf],
                    feed_dict={
                        self.obs0_norm: obs0_norm,
                        self.actions_norm: action_norm,
                        self.obs1_norm: obs1_norm,
                        self.obs2_norm: obs2_norm,
                        self.obs3_norm: obs3_norm,
                        self.obs4_norm: obs4_norm,
                        self.obs5_norm: obs5_norm
                    })
            self.dynamics_adam.update(dynamics_grads, stepsize=self.learning_rate)
        return dynamics_loss


class EnsembleForwardDynamics:
    @store_args
    def __init__(self, num_models, dimo, dimu, t, clip_norm=5, norm_eps=1e-4, hidden=256, layers=4, learning_rate=1e-3):
        self.num_models = num_models
        self.models = []
        for i in range(num_models):
            self.models.append(ForwardDynamicsNumpy(dimo, dimu, t, clip_norm, norm_eps, hidden, layers, learning_rate, name=str(i)))

    def predict_next_state(self, obs0, actions, mode='mean'):
        # random select prediciton or mean prediction
        if mode == 'random':
            idx = int(np.random.random() * self.num_models)
            model = self.models[idx]
            result = model.predict_next_state(obs0, actions)
        elif mode == 'mean':
            res = []
            for model in self.models:
                res.append(model.predict_next_state(obs0, actions))

            result_array = np.array(res).transpose([1,0,2])
            result = result_array.mean(axis=1)
        elif mode == 'mean_std':
            res = []
            for model in self.models:
                res.append(model.predict_next_state(obs0, actions))
            result_array = np.array(res).transpose([1,0,2])
            result = result_array.mean(axis=1)
            std = result_array.std(axis=1).sum(axis=1)
            return result, std
        else:
            raise NotImplementedError('No such prediction mode!')
        return result


    def update(self, obs0, actions, obs1, obs2, obs3, obs4, obs5, times=1, mode='random'):
        # update all or update a random model
        if mode == 'all':
            dynamics_per_sample_loss = []
            for model in self.models:
                loss = model.update(obs0, actions, obs1, obs2, obs3, obs4, obs5, times)
                dynamics_per_sample_loss.append(loss)
            dynamics_per_sample_loss_array = np.array(dynamics_per_sample_loss)
            dynamics_per_sample_loss = dynamics_per_sample_loss_array.mean(axis=0)
        elif mode == 'random':
            idx = int(np.random.random() * self.num_models)
            model = self.models[idx]
            dynamics_per_sample_loss = model.update(obs0, actions, obs1, obs2, obs3, obs4, obs5, times)
        else:
            raise NotImplementedError('No such update mode!')
        return dynamics_per_sample_loss
