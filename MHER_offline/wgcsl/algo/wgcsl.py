from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from wgcsl.common import logger
from wgcsl.algo.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from wgcsl.algo.normalizer import Normalizer, AverageNormNumpy, MaxNormNumpy
from wgcsl.algo.replay_buffer import ReplayBuffer
from wgcsl.common.mpi_adam import MpiAdam
from wgcsl.common import tf_util
import time


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class WGCSL(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, random_sampler, gamma,  supervised_sampler, use_supervised, su_method,
                 conservation_sampler, model_sampler, alpha, use_model_her=False, use_conservation=False, conservation_rate=1, reuse=False, offline_train=False, **kwargs):
        """Implementation of policy with value funcion that is used in combination with WGCSL
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model. 
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        if self.use_conservation:
            stage_shapes['neg_u'] = stage_shapes['u']
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key]) for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)
        # buffer_size % rollout_batch_size should be zero
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size 

        if self.use_supervised:
            sampler = self.supervised_sampler
            info = {
                'use_supervised':True,
                'gamma':self.gamma,
                'train_policy':self.train_policy,
                'get_Q_pi':self.get_Q_pi,
                'method': self.su_method,
                'get_ags_std':self.get_ags_std,
            }
        elif self.use_conservation:
            sampler = self.conservation_sampler
            info = {
                'get_Q': self.get_Q,
                'random_action_fun': self._random_action
            }
        elif self.use_model_her:
            sampler = self.model_sampler
            from wgcsl.algo.dynamics import EnsembleForwardDynamics
            self.dynamic_model = EnsembleForwardDynamics(3, self.dimo, self.dimu)
            info = {
                'dynamic_model': self.dynamic_model,
                'action_fun': self.action_only,
                'obs_to_goal': self.info['obs_to_goal'],
                'alpha': self.alpha
            }
        else: 
            sampler = self.sample_transitions
            info = {}
        
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, sampler, self.sample_transitions, info)
        self.process_rate = 0
    
    def get_ags_std(self):
        return self.buffer.ag_std_array.copy()

    def set_process(self, rate):
        self.process_rate = rate
    
    def get_process(self):
        return self.process_rate

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g, ):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'], use_target_net=use_target_net)
        return actions, None, None, None

    def action_only(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.target  #self.target if use_target_net else
        action = self.sess.run(policy.pi_tf, feed_dict={
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        })
        return action

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        if self.use_supervised:
            policy = self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    
    def get_Q(self, o, g, u):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.main
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu)
        }
        ret = self.sess.run(policy.Q_tf, feed_dict=feed)
        return ret

    def get_Q_pi(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        policy = self.main #self.target
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf:g.reshape(-1, self.dimg)
        }
        ret = self.sess.run(policy.Q_pi_tf, feed_dict=feed)
        return ret

    def get_target_Q(self, o, g, a, ag):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.main
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32) #??
        }

        ret = self.sess.run(policy.Q_tf, feed_dict=feed)
        return ret

    def store_episode(self, episode_batch, update_stats=True): #init=False
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key 'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # episode doesn't has key o_2
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            # add transitions to normalizer
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats
            # training normalizer online 
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            self.u_stats.update(transitions['u'])
            self.u_stats.recompute_stats()

    def train_policy(self, o, g, u, weights=None):
        if weights is None:
            weights = np.ones(o.shape[0])
        
        pi_sl_loss, pi_sl_grad = self.sess.run(
            [self.policy_sl_loss, self.pi_sl_grad_tf],
            feed_dict={
                self.gcsl_weight_tf: weights,
                self.main.o_tf: o,
                self.main.g_tf: g,
                self.main.u_tf : u
            }
        )
        if not self.use_supervised:
            self.pi_adam.update(pi_sl_grad, self.pi_lr)  
            for _ in range(3): 
                self.update_target_net()
        else:
            self.pi_adam.update(pi_sl_grad, self.pi_lr)
        return pi_sl_loss

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([  
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,  
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)


    def sample_batch(self, method='list'):
        transitions = self.buffer.sample(self.batch_size)   #otherwise only sample from primary buffer

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        if self.offline_train:
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

        if method == 'list':
            transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        else:
            transitions_batch = transitions
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
            self.temp_batch = batch
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        if not self.use_supervised: 
            critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
            self._update(Q_grad, pi_grad)
            return critic_loss, actor_loss
        else:
            self.update_critic_only()
    
    def update_critic_only(self):
        V, r, target_tf, main_tf, critic_loss, Q_grad = self.sess.run([  
            self.target.Q_pi_tf,
            self.batch_r,
            self.target_tf,
            self.main.Q_tf,
            self.Q_loss_tf,
            self.Q_grad_tf,
        ])
        self.Q_adam.update(Q_grad, self.Q_lr)
    

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)


    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a WGCSL agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('u_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.u_stats = Normalizer(self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        self.batch_r = batch_tf['r']
        clip_range = (-self.clip_return, self.clip_return)

        if self.su_method == 'AWR':
            target_tf = tf.clip_by_value(batch_tf['r'], *clip_range)
        else:
            target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.target_tf = target_tf  
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        if self.use_conservation:
            self.conservation_loss = tf.reduce_mean(self.main.Q_tf_neg)
            self.Q_loss_tf += self.conservation_rate * tf.reduce_mean(self.main.Q_tf_neg)

        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.temp_pi_loss = -tf.reduce_mean(self.main.Q_pi_tf) + self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.temp_action_loss = self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        # training policy with supervised learning (GCSL)
        self.gcsl_weight_tf = tf.placeholder(tf.float32, shape=(None,) , name='weights')
        self.weighted_sl_loss = tf.reduce_mean(tf.square(self.main.u_tf - self.main.pi_tf),axis=1)
        self.policy_sl_loss = tf.reduce_mean(self.gcsl_weight_tf * self.weighted_sl_loss) 

        # trainning with model-based her
        if self.use_model_her:
            self.policy_sl_loss_dim = tf.reduce_mean(tf.square(self.main.u_tf - self.main.pi_tf))  
            self.pi_loss_tf += self.alpha * self.policy_sl_loss_dim

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        pi_sl_grads_tf = tf.gradients(self.policy_sl_loss, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf) and len(self._vars('main/pi')) == len(pi_sl_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.pi_sl_grads_vars_tf = zip(pi_sl_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
        self.pi_sl_grad_tf = flatten_grads(grads=pi_sl_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')

        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        logs += [('stats_u/mean', np.mean(self.sess.run([self.u_stats.mean])))]
        logs += [('stats_u/std', np.mean(self.sess.run([self.u_stats.std])))]
        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def save(self, save_path):
        tf_util.save_variables(save_path)

