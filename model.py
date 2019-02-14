# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers
from chainer.training import updaters, Trainer, extensions

def compose(x, funcs):
	y = x
	for f in funcs:
		y = f(y)
		
	return y

class Critic(Chain):
	def __init__(self):
		super().__init__()
		kwds = {
			"ksize": 4,
			"stride": 1,
			"pad": 1
		}
		with self.init_scope():
			self.conv1 = L.Convolution2D(1, 32, **kwds)		# (14,14)
			self.conv2 = L.Convolution2D(32, 64, **kwds)		# (7,7)
			self.conv3 = L.Convolution2D(64, 128, ksize=2, stride=1, pad=0)		# (6,6)
			self.conv4 = L.Convolution2D(128, 256, **kwds)	# (3,3)
			self.conv5 = L.Convolution2D(256, 1, ksize=1, stride=1, pad=0)
			
	def __call__(self, x):
		h = compose(x, [
			self.conv1, F.leaky_relu,
			self.conv2, F.leaky_relu,
			self.conv3, F.leaky_relu,
			self.conv4, F.leaky_relu,
			self.conv5,
			lambda x:F.mean(x, axis=(1,2,3))
		])
		
		return h

		
class Generator(Chain):
	"""
	U-Net構造
	"""
	def __init__(self):
		super().__init__()
		kwds = {
			"ksize": 4,
			"stride": 2,
			"pad": 1,
			"nobias": True
		}
		with self.init_scope():
			self.conv1 = L.Convolution2D(1, 32, **kwds)		# (14,14)
			self.conv2 = L.Convolution2D(32, 64, **kwds)		# (7,7)
			self.conv3 = L.Convolution2D(64, 128, ksize=2, stride=1, pad=0, nobias=True)	# (6,6)
			self.conv4 = L.Convolution2D(128, 256, **kwds)	# (3,3)
			
			self.deconv4 = L.Deconvolution2D(256, 128, **kwds)		# (6,6)
			self.deconv3 = L.Deconvolution2D(128+128, 64, ksize=2, stride=1, pad=0, nobias=True)	# (7,7)
			self.deconv2 = L.Deconvolution2D(64+64, 32, **kwds)		# (14,14)
			self.deconv1 = L.Deconvolution2D(32+32, 1, ksize=4, stride=2, pad=1)					# (28,28)
			
			self.bn_conv1 = L.BatchNormalization(32)
			self.bn_conv2 = L.BatchNormalization(64)
			self.bn_conv3 = L.BatchNormalization(128)
			self.bn_conv4 = L.BatchNormalization(256)
			self.bn_deconv4 = L.BatchNormalization(128)
			self.bn_deconv3 = L.BatchNormalization(64)
			self.bn_deconv2 = L.BatchNormalization(32)
			
	def __call__(self, x):
		h_conv1 = compose(x, [self.conv1, self.bn_conv1, F.leaky_relu])
		h_conv2 = compose(h_conv1, [self.conv2, self.bn_conv2, F.leaky_relu])
		h_conv3 = compose(h_conv2, [self.conv3, self.bn_conv3, F.leaky_relu])
		
		h = compose(h_conv3, [
			self.conv4, self.bn_conv4, F.leaky_relu,
			self.deconv4, self.bn_deconv4, F.leaky_relu
		])
		
		h = F.concat((h, h_conv3))
		del h_conv3
		h = compose(h, [self.deconv3, self.bn_deconv3, F.leaky_relu])
		
		h = F.concat((h, h_conv2))
		del h_conv2
		h = compose(h, [self.deconv2, self.bn_deconv2, F.leaky_relu])
		
		h = F.concat((h, h_conv1))
		del h_conv1
		h = F.tanh(self.deconv1(h))
		
		return h
