"""contrastive feature learning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
sys.path.append('..')
from lib.utils import print_color



class Feature:
  def __init__(
      self,
      observation_size,
      # network properties
      fc_layer_params=(256, 256),
      graph=None,
      scope_name='feature',
      normalize_observations=False,
      ):
    self._observation_size = observation_size
    self._normalize_observations = normalize_observations

    # tensorflow requirements
    if graph is not None:
      self._graph = graph
    else:
      self._graph = tf.compat.v1.get_default_graph()
    self._scope_name = scope_name

    # network properties
    self._fc_layer_params = fc_layer_params
    self._use_placeholders = False
    self._session = None

    # saving/restoring variables
    self._saver = None


  # feature graph
  def _feature_graph(self, timesteps, pos_sample, neg_sample):
    ts_out, pos_out, neg_out = timesteps, pos_sample, neg_sample
    with tf.compat.v1.variable_scope('feature_network'):
      for idx, layer_size in enumerate(self._fc_layer_params):
        ts_out = tf.compat.v1.layers.dense(
            ts_out,
            layer_size,
            activation=tf.nn.relu,
            name='hid_' + str(idx),
            reuse=False)
        pos_out = tf.compat.v1.layers.dense(
            pos_out,
            layer_size,
            activation=tf.nn.relu,
            name='hid_' + str(idx),
            reuse=True)
        neg_out = tf.compat.v1.layers.dense(
            neg_out,
            layer_size,
            activation=tf.nn.relu,
            name='hid_' + str(idx),
            reuse=True)
      # compress feature to 2 dims
      self.ts_out = ts_out = tf.compat.v1.layers.dense(
          ts_out,
          2,
          activation=None,
          name='feature',
          reuse=False)
      self.pos_out = pos_out = tf.compat.v1.layers.dense(
          pos_out,
          2,
          activation=None,
          name='feature',
          reuse=True)
      self.neg_out = neg_out = tf.compat.v1.layers.dense(
          neg_out,
          2,
          activation=None,
          name='feature',
          reuse=True)

      min_dist = tf.reduce_mean(tf.square(ts_out-pos_out), -1)
      max_dist = 1 - tf.reduce_mean(tf.square(neg_out - ts_out), -1)
      max_dist = tf.maximum(max_dist, 0.)
      self.dist_loss = tf.reduce_mean(max_dist + min_dist)

  def _get_dict(self,
                input_data,
                target_data,
                neg_sample,
                batch_size=-1,
                batch_norm=False):
    if batch_size > 0:
      shuffled_batch = np.random.permutation(len(input_data))[:batch_size]
    else:
      shuffled_batch = np.arange(len(input_data))

    # if we are noising the input, it is better to create a new copy of the numpy arrays
    batched_input = input_data[shuffled_batch, :]
    batched_targets = target_data[shuffled_batch, :]
    batched_neg = neg_sample[shuffled_batch, :]

    return_dict = {
        self.timesteps_pl: batched_input,
        self.pos_sample_pl: batched_targets,
        self.neg_sample_pl: batched_neg,
    }
    if self._normalize_observations:
        return_dict[self.is_training_pl] = batch_norm
    return return_dict

  def make_placeholders(self):
    self._use_placeholders = True
    with self._graph.as_default(), tf.compat.v1.variable_scope(self._scope_name):
      self.timesteps_pl = tf.compat.v1.placeholder(
          tf.float32, shape=(None, self._observation_size), name='separ_timesteps_pl')
      self.pos_sample_pl = tf.compat.v1.placeholder(
          tf.float32, shape=(None, self._observation_size), name='separ_pos_pl')
      self.neg_sample_pl = tf.compat.v1.placeholder(
          tf.float32, shape=(None, self._observation_size), name='separ_neg_pl')
      if self._normalize_observations:
          self.is_training_pl = tf.compat.v1.placeholder(tf.bool, name='separ_batch_norm_pl')

  def set_session(self, session=None, initialize_or_restore_variables=False):
    if session is None:
      self._session = tf.Session(graph=self._graph)
    else:
      self._session = session

    # only initialize uninitialized variables
    if initialize_or_restore_variables:
      if tf.io.gfile.exists(self._save_prefix):
        self.restore_variables()
      with self._graph.as_default():
        var_list = tf.compat.v1.global_variables(
        ) + tf.compat.v1.local_variables()
        is_initialized = self._session.run(
            [tf.compat.v1.is_variable_initialized(v) for v in var_list])
        uninitialized_vars = []
        for flag, v in zip(is_initialized, var_list):
          if not flag:
            uninitialized_vars.append(v)

        if uninitialized_vars:
          self._session.run(
              tf.compat.v1.variables_initializer(uninitialized_vars))

  def build_graph(self,
                  timesteps=None):
    with self._graph.as_default(), tf.compat.v1.variable_scope(
        self._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
      if self._use_placeholders:
        timesteps = self.timesteps_pl
        pos_sample = self.pos_sample_pl
        neg_sample = self.neg_sample_pl
        if self._normalize_observations:
            is_training = self.is_training_pl
            self.input_norm_layer = tf.compat.v1.layers.BatchNormalization(
                scale=False, center=False, name='input_normalization')
            timesteps = self.input_norm_layer(
                timesteps, training=is_training)

            self.pos_norm_layer = tf.compat.v1.layers.BatchNormalization(
                scale=False, center=False, name='pos_normalization')
            pos_sample = self.pos_norm_layer(
                pos_sample, training=is_training)

            self.neg_norm_layer = tf.compat.v1.layers.BatchNormalization(
                scale=False, center=False, name='neg_normalization')
            neg_sample = self.neg_norm_layer(
                neg_sample, training=is_training)

      self._feature_graph(timesteps, pos_sample, neg_sample)

  def minimize_dist_op(self, learning_rate=3e-4):
    with self._graph.as_default():
      update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
          self.min_dist_op = tf.compat.v1.train.AdamOptimizer(
              learning_rate=learning_rate,
              name='adam_max').minimize(self.dist_loss)
          print_color("Optimize slow feature !!!")
          return self.min_dist_op

  def create_saver(self, save_prefix):
    if self._saver is not None:
      return self._saver
    else:
      with self._graph.as_default():
        self._variable_list = {}
        for var in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self._scope_name):
          self._variable_list[var.name] = var
        self._saver = tf.compat.v1.train.Saver(
            self._variable_list, save_relative_paths=True,
            max_to_keep=1000)
        self._save_prefix = save_prefix

  def save_variables(self, global_step):
    if not tf.io.gfile.exists(self._save_prefix):
      tf.io.gfile.makedirs(self._save_prefix)

    self._saver.save(
        self._session,
        os.path.join(self._save_prefix, 'ckpt'),
        global_step=global_step)

  def restore_variables(self):
    self._saver.restore(self._session,
                        tf.compat.v1.train.latest_checkpoint(self._save_prefix))

  # all functions here-on require placeholders----------------------------------
  def train(self,
            timesteps,
            next_timesteps,
            neg_sample,
            batch_size=512,
            num_steps=1):
    if not self._use_placeholders:
      return

    loss_log = []
    for _ in range(num_steps):
      _, dist_loss = self._session.run(
          [self.min_dist_op, self.dist_loss],
          feed_dict=self._get_dict(
              timesteps,
              next_timesteps,
              neg_sample,
              batch_size=batch_size,
              batch_norm=True))
      loss_log.append(dist_loss)
    return np.mean(loss_log)

  def eval_phi(self, timesteps):
      if self._normalize_observations:
          return self._session.run([self.ts_out, self.neg_out],
                                   feed_dict={self.timesteps_pl: timesteps,
                                              self.neg_sample_pl: timesteps,
                                              self.is_training_pl:False})
      else:
          return self._session.run([self.ts_out, self.neg_out],
                                   feed_dict={self.timesteps_pl: timesteps,
                                              self.neg_sample_pl: timesteps,
                                             })


