# coding: utf-8

import numpy as np
import cv2

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, iterators, Chain, optimizers
from chainer.training import updaters, Trainer, extensions

import matplotlib
matplotlib.use("Agg")

import pprint
from pathlib import Path

from model import Critic, Generator

def parse_args():
	import argparse

	desc = """
	Visual Feature Attribution using Wasserstein GANs, CVPR 2018
	"""
	
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument("-b", "--batchsize", type=int, default=100)
	parser.add_argument("-e", "--epoch", type=int, default=100)
	parser.add_argument("--alpha", type=float, default=0.0001, help="alpha of Adam optimizer")
	parser.add_argument("--beta1", type=float, default=0, help="beta1 of Adam")
	parser.add_argument("--beta2", type=float, default=0.9, help="beta2 of Adam")
	parser.add_argument("--n_cri1", type=int, default=5)
	parser.add_argument("--n_cri2", type=int, default=100)
	parser.add_argument("--gp_lam", type=float, default=10.0, help="weight of gradient penalty (WGAN-GP)")
	parser.add_argument("--l1_lam", type=float, default=0.01, help="L1 loss of generator")
	parser.add_argument("-g", type=int, default=0, help="GPU ID (negative value indicates CPU mode)")
	parser.add_argument("--result_dir", default="result")
	parser.add_argument("--neg_numbers", type=int, nargs="*", default=[3], help="digits regarded as negative example")
	parser.add_argument("--pos_numbers", type=int, nargs="*", default=[8], help="digits regarded as positive example")
	args = parser.parse_args()
	
	pprint.pprint(vars(args))
	main(args)
	
def main(args):
	chainer.config.user_gpu = args.g
	if args.g >= 0:
		chainer.backends.cuda.get_device_from_id(args.g).use()
		print("GPU mode")

	mnist_3 = get_mnist_num(args.neg_numbers)
	mnist_8 = get_mnist_num(args.pos_numbers)
	
	# iteratorを作成
	kwds = {
		"batch_size": args.batchsize,
		"shuffle": True,
		"repeat": True
	}
	mnist_3_iter = iterators.SerialIterator(mnist_3, **kwds)
	mnist_8_iter = iterators.SerialIterator(mnist_8, **kwds)
	
	generator = Generator()
	critic = Critic()
	if args.g >= 0:
		generator.to_gpu()
		critic.to_gpu()
	
	adam_args = args.alpha, args.beta1, args.beta2
	opt_g = optimizers.Adam(*adam_args)
	opt_g.setup(generator)
	opt_c = optimizers.Adam(*adam_args)
	opt_c.setup(critic)
	
	updater = WGANUpdater(mnist_3_iter, mnist_8_iter, opt_g, opt_c, args.n_cri1, args.n_cri2, args.gp_lam, args.l1_lam)
	trainer = Trainer(updater, (args.epoch,"epoch"), out=args.result_dir)
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport(["epoch", "generator/loss", "critic/loss"]))
	trainer.extend(extensions.ProgressBar())
	trainer.extend(extensions.PlotReport(("generator/loss","critic/loss"),"epoch", file_name="loss_plot.eps"))
	trainer.extend(ext_save_img(generator, mnist_8, args.result_dir+"/out_images"))
	
	trainer.run()

def get_mnist_num(dig_list:list) -> np.ndarray:
	mnist_dataset = chainer.datasets.get_mnist()[0]	# MNISTデータ取得
	mnist_dataset = [img for img,label in mnist_dataset[:] if label in dig_list]
	mnist_dataset = np.array(mnist_dataset, dtype=np.float32)
	mnist_dataset = mnist_dataset.reshape((-1,1,28,28))
	return mnist_dataset
	
# 生成画像を保存するextension
def ext_save_img(generator, pos_data, out):
	out_path = Path(out)
	try:
		out_path.mkdir(parents=True)
	except FileExistsError:
		pass

	@chainer.training.make_extension(trigger=(1,"epoch"))
	def _ext_save_img(trainer):
		i = np.random.randint(len(pos_data))
		img = np.expand_dims(pos_data[i], axis=0)
		if chainer.config.user_gpu >= 0:
			img = generator.xp.asarray(img)
		with chainer.using_config("train", False):
			m = generator(Variable(img)).array.reshape((28,28))
		if chainer.config.user_gpu >= 0:
			m = generator.xp.asnumpy(m)
		m_col = 255.0*(m+1)/2
		m_col = cv2.applyColorMap(m_col.astype(np.uint8), cv2.COLORMAP_JET)
		
		orig_img = pos_data[i].reshape((28,28))
		img = (np.clip(orig_img+m, 0, 1)*255).astype(np.uint8)
		orig_img = (orig_img*255).astype(np.uint8)

		cv2.imwrite(str(out_path/"m_epoch_{:04d}.png".format(trainer.updater.epoch)), m_col)
		cv2.imwrite(str(out_path/"x_epoch_{:04d}.png".format(trainer.updater.epoch)), img)
		cv2.imwrite(str(out_path/"o_epoch_{:04d}.png".format(trainer.updater.epoch)), orig_img)

	return _ext_save_img

	
	
class WGANUpdater(updaters.StandardUpdater):
	def __init__(self, neg_iterator, pos_iterator, gen_opt, cri_opt, n_cri1, n_cri2, gp_lam, l1_lam, **kwds):
		opts = {
			"gen":gen_opt,
			"cri":cri_opt
		}
		iters = {
			"main":neg_iterator,
			"neg_iter":neg_iterator,
			"pos_iter":pos_iterator
		}
		self.n_cri1 = n_cri1
		self.n_cri2 = n_cri2
		self.gp_lam = gp_lam
		self.l1_lam = l1_lam
		super().__init__(iters, opts, **kwds)
		
	def update_core(self):
		gen_opt = self.get_optimizer("gen")
		cri_opt = self.get_optimizer("cri")
		generator = gen_opt.target
		critic = cri_opt.target
		batch_size = self.get_iterator("main").batch_size
		
		# バッチ（負例）を取得
		x_real = self.get_iterator("neg_iter").next()
		x_real = Variable(np.stack(x_real))
		if chainer.config.user_gpu >= 0:
			x_real.to_gpu()
		
		xp = x_real.xp
		
		# update critic
		cri_upd_num = self.n_cri2 if self.iteration <= 25 or self.iteration%500 == 0 else self.n_cri1
		for i in range(cri_upd_num):
			x_pos = self.get_iterator("pos_iter").next()
			x_pos = Variable(np.stack(x_pos))
			if chainer.config.user_gpu >= 0:
				x_pos.to_gpu()
			m = generator(x_pos)
			x_fake = x_pos+m
			
			cri_loss = F.average(critic(x_fake)-critic(x_real))	# Wasserstein距離の逆符号
			
			# gradient penalty
			eps = xp.random.uniform(size=(batch_size,1,1,1)).astype(np.float32)
			x_fusion = eps*x_real + (1-eps)*x_fake	# (N,1,H,W)
			g_critic = chainer.grad([critic(x_fusion)], [x_fusion], enable_double_backprop=True)[0]	# (N,1,H,W)
			gp = F.batch_l2_norm_squared(g_critic)
			gp = F.average((F.sqrt(gp)-1)**2)
			total_loss = cri_loss + self.gp_lam*gp
			
			critic.cleargrads()
			total_loss.backward()
			cri_opt.update()
			
		# update generator
		x_pos = self.get_iterator("pos_iter").next()
		x_pos = Variable(np.stack(x_pos))
		if chainer.config.user_gpu >= 0:
			x_pos.to_gpu()
		m = generator(x_pos)
		l1_loss = F.sum(F.absolute(m))/batch_size	# L1 norm
		x_fake = x_pos+m
		gen_loss = -F.average(critic(x_fake)) + self.l1_lam*l1_loss
		
		generator.cleargrads()
		critic.cleargrads()
		gen_loss.backward()
		gen_opt.update()
		
		chainer.report({
			"generator/loss":gen_loss,
			"critic/loss":cri_loss,
			"main/wdist":-cri_loss
		})
		
if __name__ == "__main__":
	parse_args()
