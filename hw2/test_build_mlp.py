import unittest
import random
from train_pg_f18 import *

class TestBuildMLP(unittest.TestCase):
	def setUp(self):
		self.sess = tf.Session()
		self.output_size = random.randint(5, 15)
		self.n_layers = 3
		self.size = 15
		self.input_shape = (None, 12)
		self.scope = 'test-scope-%d' % self.output_size
		self.input_placeholder = tf.placeholder(tf.float32, shape=self.input_shape)
		self.output_placeholder = build_mlp(self.input_placeholder, 
			self.output_size, 
			self.scope, 
			self.n_layers, 
			self.size, 
			activation=tf.tanh, 
			output_activation=None)

	def test_output_tensor(self):
		self.assertEqual(self.output_placeholder.get_shape().as_list()[1], self.output_size)
		self.assertEqual(self.output_placeholder.name, '%s/output/BiasAdd:0' % self.scope)
