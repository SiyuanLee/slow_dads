# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamics Prediction and Training."""

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


# TODO(architsh): Implement the dynamics with last K step input
class SkillDynamics:

    def __init__(
            self,
            observation_size,
            action_size,
            restrict_observation=0,
            normalize_observations=False,
            # network properties
            fc_layer_params=(256, 256),
            network_type='default',
            num_components=1,
            fix_variance=False,
            reweigh_batches=False,
            graph=None,
            scope_name='skill_dynamics',
            learn_slow_feature=False,
            loss_coeff=0.1,
            learn_feature_separ=False):

        self._observation_size = observation_size
        self._action_size = action_size
        self._normalize_observations = normalize_observations
        self._restrict_observation = restrict_observation
        self._reweigh_batches = reweigh_batches
        self._learn_slow_feature = learn_slow_feature
        self.loss_coeff = loss_coeff
        self._learn_feature_separ = learn_feature_separ
        if learn_slow_feature or learn_feature_separ:
            self.predict_dim = 2
        else:
            self.predict_dim = observation_size

        # tensorflow requirements
        if graph is not None:
            self._graph = graph
        else:
            self._graph = tf.compat.v1.get_default_graph()
        self._scope_name = scope_name

        # dynamics network properties
        self._fc_layer_params = fc_layer_params
        self._network_type = network_type
        self._num_components = num_components
        self._fix_variance = fix_variance
        if not self._fix_variance:
            self._std_lower_clip = 0.3
            self._std_upper_clip = 10.0

        self._use_placeholders = False
        self.log_probability = None
        self.dyn_max_op = None
        self.dyn_min_op = None
        self._session = None
        self._use_modal_mean = False

        # saving/restoring variables
        self._saver = None

    def _get_distribution(self, out, reuse=False):
        if self._num_components > 1:
            self.logits = tf.compat.v1.layers.dense(
                out, self._num_components, name='logits', reuse=reuse)
            means, scale_diags = [], []
            for component_id in range(self._num_components):
                means.append(
                    tf.compat.v1.layers.dense(
                        out,
                        self.predict_dim,
                        name='mean_' + str(component_id),
                        reuse=reuse))
                if not self._fix_variance:
                    scale_diags.append(
                        tf.clip_by_value(
                            tf.compat.v1.layers.dense(
                                out,
                                self.predict_dim,
                                activation=tf.nn.softplus,
                                name='stddev_' + str(component_id),
                                reuse=reuse), self._std_lower_clip,
                            self._std_upper_clip))
                else:
                    scale_diags.append(
                        tf.fill([tf.shape(out)[0], self.predict_dim], 1.0))

            self.means = tf.stack(means, axis=1)
            self.scale_diags = tf.stack(scale_diags, axis=1)
            return tfp.distributions.MixtureSameFamily(
                mixture_distribution=tfp.distributions.Categorical(
                    logits=self.logits),
                components_distribution=tfp.distributions.MultivariateNormalDiag(
                    loc=self.means, scale_diag=self.scale_diags))

        else:
            mean = tf.compat.v1.layers.dense(
                out, self._observation_size, name='mean', reuse=reuse)
            if not self._fix_variance:
                stddev = tf.clip_by_value(
                    tf.compat.v1.layers.dense(
                        out,
                        self._observation_size,
                        activation=tf.nn.softplus,
                        name='stddev',
                        reuse=reuse), self._std_lower_clip,
                    self._std_upper_clip)
            else:
                stddev = tf.fill([tf.shape(out)[0], self._observation_size], 1.0)
            return tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=stddev)

    # dynamics graph with separate pipeline for skills and timesteps
    def _graph_with_separate_skill_pipe(self, timesteps, actions, pos_sample, neg_sample, next_timesteps):
        skill_out = actions
        with tf.compat.v1.variable_scope('action_pipe'):
            for idx, layer_size in enumerate((self._fc_layer_params[0] // 2,)):
                skill_out = tf.compat.v1.layers.dense(
                    skill_out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=tf.compat.v1.AUTO_REUSE)

        ts_out, next_ts, pos_out, neg_out = timesteps, next_timesteps, pos_sample, neg_sample
        with tf.compat.v1.variable_scope('ts_pipe'):
            for idx, layer_size in enumerate((self._fc_layer_params[0] // 2,)):
                ts_out = tf.compat.v1.layers.dense(
                    ts_out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=tf.compat.v1.AUTO_REUSE)
                pos_out = tf.compat.v1.layers.dense(
                    pos_out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=tf.compat.v1.AUTO_REUSE)
                neg_out = tf.compat.v1.layers.dense(
                    neg_out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=tf.compat.v1.AUTO_REUSE)
                next_ts = tf.compat.v1.layers.dense(
                    next_ts,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=tf.compat.v1.AUTO_REUSE)
                # stop gradient for target feature
                next_ts = tf.stop_gradient(next_ts)
            # compress feature to 2 dims
            self.ts_out = ts_out = tf.compat.v1.layers.dense(
                ts_out,
                2,
                activation=None,
                name='feature',
                reuse=tf.compat.v1.AUTO_REUSE)
            self.pos_out = pos_out = tf.compat.v1.layers.dense(
                pos_out,
                2,
                activation=None,
                name='feature',
                reuse=tf.compat.v1.AUTO_REUSE)
            self.neg_out = neg_out = tf.compat.v1.layers.dense(
                neg_out,
                2,
                activation=None,
                name='feature',
                reuse=tf.compat.v1.AUTO_REUSE)
            self.next_ts = next_ts = tf.compat.v1.layers.dense(
                next_ts,
                2,
                activation=None,
                name='feature',
                reuse=tf.compat.v1.AUTO_REUSE)
            # stop gradient for target feature
            next_ts = tf.stop_gradient(next_ts)
            min_dist = tf.reduce_mean(tf.square(ts_out - pos_out), -1)
            max_dist = 1 - tf.reduce_mean(tf.square(neg_out - ts_out), -1)
            max_dist = tf.maximum(max_dist, 0.)
            dist_loss = tf.reduce_mean(max_dist + min_dist)
            print_color("dist loss:")
            print(dist_loss)

        # out = tf.compat.v1.layers.flatten(tf.einsum('ai,aj->aij', ts_out, skill_out))
        out = tf.concat([ts_out, skill_out], axis=1)
        with tf.compat.v1.variable_scope('joint'):
            for idx, layer_size in enumerate(self._fc_layer_params[1:]):
                out = tf.compat.v1.layers.dense(
                    out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=tf.compat.v1.AUTO_REUSE)

        print_color("use seperate network !!!")
        return self._get_distribution(out), dist_loss, next_ts

    def _get_compressed_feature(self, timesteps, next_timesteps, feature):
        ts_out, next_ts = timesteps, next_timesteps
        with tf.compat.v1.variable_scope('feature_network'):
            for idx, layer_size in enumerate(feature._fc_layer_params):
                ts_out = tf.compat.v1.layers.dense(
                    ts_out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=True)
                next_ts = tf.compat.v1.layers.dense(
                    next_ts,
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
                reuse=True)
            next_ts = tf.compat.v1.layers.dense(
                next_ts,
                2,
                activation=None,
                name='feature',
                reuse=True)
            # stop gradient from feature network
            next_ts = tf.stop_gradient(next_ts)
            ts_out = tf.stop_gradient(ts_out)
            return ts_out, next_ts

    # reuse weights from feature network
    def _graph_reuse(self, ts_out, actions, reuse=False):
        skill_out = actions
        with tf.compat.v1.variable_scope('action_pipe'):
            for idx, layer_size in enumerate((self._fc_layer_params[0] // 2,)):
                skill_out = tf.compat.v1.layers.dense(
                    skill_out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=reuse)

        # out = tf.compat.v1.layers.flatten(tf.einsum('ai,aj->aij', ts_out, skill_out))
        out = tf.concat([ts_out, skill_out], axis=1)
        with tf.compat.v1.variable_scope('joint'):
            for idx, layer_size in enumerate(self._fc_layer_params[1:]):
                out = tf.compat.v1.layers.dense(
                    out,
                    layer_size,
                    activation=tf.nn.relu,
                    name='hid_' + str(idx),
                    reuse=reuse)

        print_color("use seperate network !!!")
        return self._get_distribution(out, reuse)

    # simple dynamics graph
    def _default_graph(self, timesteps, actions):
        out = tf.concat([timesteps, actions], axis=1)
        for idx, layer_size in enumerate(self._fc_layer_params):
            out = tf.compat.v1.layers.dense(
                out,
                layer_size,
                activation=tf.nn.relu,
                name='hid_' + str(idx),
                reuse=tf.compat.v1.AUTO_REUSE)

        return self._get_distribution(out)

    def _get_dict(self,
                  input_data,
                  input_actions,
                  target_data,
                  neg_sample=None,
                  batch_size=-1,
                  batch_weights=None,
                  batch_norm=False,
                  noise_targets=False,
                  noise_std=0.5,
                  feature=None):
        if batch_size > 0:
            shuffled_batch = np.random.permutation(len(input_data))[:batch_size]
        else:
            shuffled_batch = np.arange(len(input_data))

        # if we are noising the input, it is better to create a new copy of the numpy arrays
        batched_input = input_data[shuffled_batch, :]
        batched_skills = input_actions[shuffled_batch, :]
        batched_targets = target_data[shuffled_batch, :]
        if neg_sample is not None:
            batched_neg = neg_sample[shuffled_batch, :]
        else:
            batched_neg = batched_input

        if self._reweigh_batches and batch_weights is not None:
            example_weights = batch_weights[shuffled_batch]

        if noise_targets:
            batched_targets += np.random.randn(*batched_targets.shape) * noise_std

        return_dict = {
            self.timesteps_pl: batched_input,
            self.actions_pl: batched_skills,
            self.next_timesteps_pl: batched_targets,
        }
        if self._normalize_observations:
            return_dict[self.is_training_pl] = batch_norm
            if feature is not None:
                return_dict[feature.neg_sample_pl] = batched_targets
                return_dict[feature.pos_sample_pl] = batched_targets
                return_dict[feature.timesteps_pl] = batched_input
                return_dict[feature.is_training_pl] = batch_norm
        if self._reweigh_batches and batch_weights is not None:
            return_dict[self.batch_weights] = example_weights
        if self._learn_slow_feature:
            return_dict[self.pos_sample_pl] = batched_targets
            return_dict[self.neg_sample_pl] = batched_neg
        return return_dict

    def _get_run_dict(self, input_data, input_actions):
        return_dict = {
            self.timesteps_pl: input_data,
            self.actions_pl: input_actions
        }
        if self._normalize_observations:
            return_dict[self.is_training_pl] = False

        return return_dict

    def make_placeholders(self):
        self._use_placeholders = True
        with self._graph.as_default(), tf.compat.v1.variable_scope(self._scope_name):
            self.timesteps_pl = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self._observation_size), name='timesteps_pl')
            self.pos_sample_pl = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self._observation_size), name='pos_pl')
            self.neg_sample_pl = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self._observation_size), name='neg_pl')
            self.actions_pl = tf.compat.v1.placeholder(
                tf.float32, shape=(None, self._action_size), name='actions_pl')
            self.next_timesteps_pl = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, self._observation_size),
                name='next_timesteps_pl')
            self.latent_ts_pl = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 2),
                name='latent_ts_pl')
            if self._normalize_observations:
                self.is_training_pl = tf.compat.v1.placeholder(tf.bool, name='batch_norm_pl')
            if self._reweigh_batches:
                self.batch_weights = tf.compat.v1.placeholder(
                    tf.float32, shape=(None,), name='importance_sampled_weights')

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
                    timesteps=None,
                    actions=None,
                    next_timesteps=None,
                    is_training=None,
                    feature=None):
        with self._graph.as_default(), tf.compat.v1.variable_scope(
                self._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            if self._use_placeholders:
                timesteps = self.timesteps_pl
                pos_sample = self.pos_sample_pl
                neg_sample = self.neg_sample_pl
                actions = self.actions_pl
                next_timesteps = self.next_timesteps_pl
                latent_ts = self.latent_ts_pl
                if self._normalize_observations:
                    is_training = self.is_training_pl

            # predict deltas instead of observations
            next_timesteps -= timesteps
            # print_color("Predict absolute next state !!!")

            if (self._restrict_observation > 0) and (not self._learn_feature_separ) and (not self._learn_slow_feature):
              timesteps = timesteps[:, self._restrict_observation:]
            else:
                print_color("NOT clip observation !!!")

            if self._normalize_observations:
                if self._learn_feature_separ:
                    timesteps = feature.input_norm_layer(
                        timesteps, training=is_training)
                    next_timesteps = feature.pos_norm_layer(
                        next_timesteps, training=is_training)
                else:
                    input_norm_layer = tf.compat.v1.layers.BatchNormalization(
                        scale=False, center=False, name='input_normalization')
                    timesteps = input_norm_layer(
                        timesteps, training=is_training)

                    pos_norm_layer = tf.compat.v1.layers.BatchNormalization(
                        scale=False, center=False, name='pos_normalization')
                    pos_sample = pos_norm_layer(
                        pos_sample, training=is_training)

                    neg_norm_layer = tf.compat.v1.layers.BatchNormalization(
                        scale=False, center=False, name='neg_normalization')
                    neg_sample = neg_norm_layer(
                        neg_sample, training=is_training)

                    self.output_norm_layer = tf.compat.v1.layers.BatchNormalization(
                        scale=False, center=False, name='output_normalization')
                    next_timesteps = self.output_norm_layer(
                        next_timesteps, training=is_training)


        if self._learn_feature_separ:
            with self._graph.as_default(), tf.compat.v1.variable_scope(
                    feature._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                timesteps, next_timesteps = self._get_compressed_feature(timesteps, next_timesteps, feature)
        with self._graph.as_default(), tf.compat.v1.variable_scope(
                self._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            if self._network_type == 'default':
                self.base_distribution = self._default_graph(timesteps, actions)
            elif self._network_type == 'separate':
                if self._learn_feature_separ:
                    self.base_distribution = self._graph_reuse(
                        timesteps, actions, False)
                    self.predict_distribution = self._graph_reuse(latent_ts, actions, True)
                    self.predict_mean = self.predict_distribution.mean()
                else:
                    self.base_distribution, self.dist_loss, next_timesteps = self._graph_with_separate_skill_pipe(
                        timesteps, actions, pos_sample, neg_sample, next_timesteps)

            # if building multiple times, be careful about which log_prob you are optimizing
            self.log_probability = self.base_distribution.log_prob(next_timesteps)
            self.mean = self.base_distribution.mean()

            return self.log_probability

    def increase_prob_op(self, learning_rate=3e-4, weights=None):
        with self._graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self._reweigh_batches:
                    self.dyn_max_op = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=learning_rate,
                        name='adam_max').minimize(-tf.reduce_mean(self.log_probability *
                                                                  self.batch_weights))
                elif weights is not None:
                    self.dyn_max_op = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=learning_rate,
                        name='adam_max').minimize(-tf.reduce_mean(self.log_probability *
                                                                  weights))
                else:
                    if self._learn_slow_feature:
                        self.dyn_max_op = tf.compat.v1.train.AdamOptimizer(
                            learning_rate=learning_rate,
                            name='adam_max').minimize(
                            -tf.reduce_mean(self.log_probability) + self.loss_coeff * self.dist_loss)
                        print_color("Optmize slow feature and log prob !!!")
                    else:
                        self.dyn_max_op = tf.compat.v1.train.AdamOptimizer(
                            learning_rate=learning_rate,
                            name='adam_max').minimize(-tf.reduce_mean(self.log_probability))

                return self.dyn_max_op

    def decrease_prob_op(self, learning_rate=3e-4, weights=None):
        with self._graph.as_default():
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self._reweigh_batches:
                    self.dyn_min_op = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=learning_rate, name='adam_min').minimize(
                        tf.reduce_mean(self.log_probability * self.batch_weights))
                elif weights is not None:
                    self.dyn_min_op = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=learning_rate, name='adam_min').minimize(
                        tf.reduce_mean(self.log_probability * weights))
                else:
                    self.dyn_min_op = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=learning_rate,
                        name='adam_min').minimize(tf.reduce_mean(self.log_probability))
                return self.dyn_min_op

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
              actions,
              next_timesteps,
              neg_sample,
              batch_weights=None,
              batch_size=512,
              num_steps=1,
              increase_probs=True,
              feature=None):
        if not self._use_placeholders:
            return

        if increase_probs:
            run_op = self.dyn_max_op
        else:
            run_op = self.dyn_min_op

        loss_log = []
        for _ in range(num_steps):
            if self._learn_slow_feature:
                _, dist_loss = self._session.run(
                    [run_op, self.dist_loss],
                    feed_dict=self._get_dict(
                        timesteps,
                        actions,
                        next_timesteps,
                        neg_sample,
                        batch_weights=batch_weights,
                        batch_size=batch_size,
                        batch_norm=True))
            else:
                self._session.run(
                    run_op,
                    feed_dict=self._get_dict(
                        timesteps,
                        actions,
                        next_timesteps,
                        batch_weights=batch_weights,
                        batch_size=batch_size,
                        batch_norm=True,
                        feature=feature))
                dist_loss = 0.
            loss_log.append(dist_loss)
        return np.mean(loss_log)

    def get_log_prob(self, timesteps, actions, next_timesteps):
        if not self._use_placeholders:
            return

        return self._session.run(
            self.log_probability,
            feed_dict=self._get_dict(
                timesteps, actions, next_timesteps, batch_norm=False))

    def eval_phi(self, timesteps):
        if self._normalize_observations:
            return self._session.run([self.ts_out],
                                     feed_dict={self.timesteps_pl: timesteps,
                                                self.neg_sample_pl: timesteps,
                                                self.is_training_pl: False})
        else:
            return self._session.run([self.ts_out],
                                     feed_dict={self.timesteps_pl: timesteps,
                                                self.neg_sample_pl: timesteps,
                                                })

    def predict_state(self, timesteps, actions):
        if not self._use_placeholders:
            return

        if self._use_modal_mean:
            all_means, modal_mean_indices = self._session.run(
                [self.means, tf.argmax(self.logits, axis=1)],
                feed_dict=self._get_run_dict(timesteps, actions))
            pred_state = all_means[[
                np.arange(all_means.shape[0]), modal_mean_indices
            ]]
        else:
            pred_state = self._session.run(
                self.mean, feed_dict=self._get_run_dict(timesteps, actions))

        if self._normalize_observations:
            with self._session.as_default(), self._graph.as_default():
                mean_correction, variance_correction = self.output_norm_layer.get_weights(
                )

            pred_state = pred_state * np.sqrt(variance_correction +
                                              1e-3) + mean_correction

        pred_state += timesteps
        return pred_state

    def predict_phi(self, latent_ts, actions):
        pred_state = self._session.run(
            self.predict_mean, feed_dict={
            self.latent_ts_pl: latent_ts,
            self.actions_pl: actions
        })
        pred_state += latent_ts
        return pred_state



