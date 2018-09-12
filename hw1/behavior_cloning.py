#!/usr/bin/env python  
"""  
hw1 
@author: Zichen Wang

Section 2. Behavioral Cloning 

Run behavioral cloning (BC) and report results on two tasks – one task where a
behavioral cloning agent achieves comparable performance to the expert, and one
task where it does not. When providing results, report the mean and standard
deviation of the return over multiple rollouts in a table, and state which task
was used. Be sure to set up a fair comparison, in terms of network size, amount
of data, and number of training iterations, and provide these details (and any
others you feel are appropriate) in the table caption. 
"""

import os
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
		self.sess = tf.Session(graph=self.graph)
		self.sess.__enter__() # equivalent to `with self.sess:`
		tf.global_variables_initializer().run() #pylint: disable=E1101

	def build_computation_graph(self):
		self.graph = tf.Graph()		
		with self.graph.as_default():
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

	def fit(self, observations, actions):
		# Fit the model with (observations, actions) data
		loss_value, _, = self.sess.run([self.loss, self.optimizer],
			feed_dict={
				self.observations: observations, 
				self.actions: actions
			})
		return loss_value

	def predict(self, observations):
		# Make predictions of actions given predictions
		return self.sess.run(self.predicted_actions, 
			feed_dict={self.observations: observations})


