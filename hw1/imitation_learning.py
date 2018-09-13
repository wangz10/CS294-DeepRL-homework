#!/usr/bin/env python  
"""  
hw1 
@author: Zichen Wang

Implementation of Imitation Learning models including:
- Behavioral Cloning 
- DAgger
"""

import os
import sys
import pickle
import tensorflow as tf
import numpy as np
import tf_util


class BehavioralCloning(object):
	"""docstring for BehavioralCloning"""
	def __init__(self, obs_dim, action_dim, n_hidden_units=None,
		lr=0.001):
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.n_hidden_units = n_hidden_units
		self.lr = lr

		self.build_computation_graph()
		self.init_tf_sess()

	def init_tf_sess(self):
		# self.sess = tf.Session(graph=self.graph)
		self.sess = tf.Session()
		self.sess.__enter__() # equivalent to `with self.sess:`
		tf.global_variables_initializer().run() #pylint: disable=E1101

	def build_computation_graph(self):
		# self.graph = tf.Graph()
		# with self.graph.as_default():
		# Set up placehoders
		self.observations = tf.placeholder(tf.float32, [None, self.obs_dim],
			name='observations')
		self.actions = tf.placeholder(tf.float32, [None, self.action_dim],
			name='actions')

		# Init variables
		with tf.variable_scope('layer1'):
			self.W1 = tf.get_variable('W', [self.obs_dim, self.n_hidden_units],
				initializer=tf.contrib.layers.xavier_initializer()
				)
			self.b1 = tf.get_variable('b', [self.n_hidden_units], 
				initializer=tf.constant_initializer(0.0))
		with tf.variable_scope('layer2'):
			self.W2 = tf.get_variable('W', [self.n_hidden_units, self.action_dim],
				initializer=tf.contrib.layers.xavier_initializer()
				)
			self.b2 = tf.get_variable('b', [self.action_dim], 
				initializer=tf.constant_initializer(0.0))

		# Computation: simple fully connected NN with one hidden layer
		self.hidden = tf.nn.relu(tf.add(
			tf.matmul(self.observations, self.W1),
			self.b1
			)
		)
		self.predicted_actions = tf.add(tf.matmul(self.hidden, self.W2), self.b2)

		self.loss = tf.reduce_mean(
			tf.squared_difference(self.actions, self.predicted_actions)
			)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

	def partial_fit(self, observations, actions):
		'''Fit the model with (observations, actions) data'''
		loss_value, _, = self.sess.run([self.loss, self.optimizer],
			feed_dict={
				self.observations: observations, 
				self.actions: actions
			})
		return loss_value

	def predict(self, observations):
		'''Make predictions of actions given predictions'''
		return self.sess.run(self.predicted_actions, 
			feed_dict={self.observations: observations})

	def train(self, observations, actions, n_epochs=10, batch_size=32, shuffle=True):
		'''Train the model using mini-batch training.
		'''
		n_samples = observations.shape[0]
		n_batches = n_samples // batch_size
		sample_idx = np.arange(n_samples)

		for i in range(n_epochs):
			for ii in range(n_batches):
				start_idx = ii*batch_size
				end_idx = (ii+1)*batch_size
				x_batch = observations[start_idx:end_idx]
				y_batch = actions[start_idx:end_idx]
				loss = self.partial_fit(x_batch, y_batch)

			print('Epoch %d loss = %.6f' % (i, loss))
			if shuffle:
				# shuffle samples
				np.random.shuffle(sample_idx)
				observations = observations[sample_idx]
				actions = actions[sample_idx]

	def evaluate_policy(self, env, n_episodes=20, max_steps=1000):
		'''evaluate current policy using env.
		'''
		returns = []
		for i in range(n_episodes):
			sys.stdout.flush()
			print('\riter {}/{}'.format(i, n_episodes), end="")
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				action = self.predict(obs.reshape(1, -1))
				obs, r, done, _ = env.step(action)
				totalr += r
				steps += 1

				if steps >= max_steps:
					break
			returns.append(totalr)
		return returns 
		

class DAgger(BehavioralCloning):
	"""docstring for DAgger"""
	def __init__(self, obs_dim, action_dim, n_hidden_units=None,
		lr=0.001):
		BehavioralCloning.__init__(self, obs_dim, action_dim, 
			n_hidden_units=n_hidden_units,
			lr=lr)
			
	# def fit(self, observations, actions):
		'''train policy pi(a_t|o_t) from human data or aggregated data
		'''

	def run_policy(self, env):
		'''run policy on the env to get dataset D_pi
		'''
		obs = env.reset()
		done = False
		D_pi = [obs]
		while not done:
			action = self.predict(obs.reshape(1, -1))
			obs, r, done, _ = env.step(action)
			D_pi.append(obs)
		return np.array(D_pi)

	def request_labels(self, D_pi, expert_policy_fn):
		'''ask human/expert to label D_pi with actions a_t
		'''
		labeled_actions = [expert_policy_fn(obs.reshape(1, -1)) for obs in D_pi]
		return np.array(labeled_actions)

	def run_policy_and_get_labeled_actions(self, env, expert_policy_fn):
		D_pi = self.run_policy(env)
		labeled_actions = self.request_labels(D_pi, expert_policy_fn)
		return D_pi, labeled_actions

