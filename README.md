# Visual Feature Attribution using Wasserstein GANs
[Baumgartner, Christian F., et al. "Visual feature attribution using Wasserstein GANs." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Baumgartner_Visual_Feature_Attribution_CVPR_2018_paper.html)

Implemented with Chainer

## Requirements
Chainer, OpenCV

```bash
$ pip install chainer opencv-python
```

## How to run
```bash
$ python vagan.py [options]
```

You can read help with `-h` option.

```bash
$ python vagan.py -h
usage: vagan.py [-h] [-b BATCHSIZE] [-e EPOCH] [--alpha ALPHA] [--beta1 BETA1]
                [--beta2 BETA2] [--n_cri1 N_CRI1] [--n_cri2 N_CRI2]
                [--gp_lam GP_LAM] [--l1_lam L1_LAM] [-g G]
                [--result_dir RESULT_DIR]
                [--neg_numbers [NEG_NUMBERS [NEG_NUMBERS ...]]]
                [--pos_numbers [POS_NUMBERS [POS_NUMBERS ...]]]

Visual Feature Attribution using Wasserstein GANs, CVPR 2018

optional arguments:
  -h, --help            show this help message and exit
  -b BATCHSIZE, --batchsize BATCHSIZE
  -e EPOCH, --epoch EPOCH
  --alpha ALPHA         alpha of Adam optimizer
  --beta1 BETA1         beta1 of Adam
  --beta2 BETA2         beta2 of Adam
  --n_cri1 N_CRI1
  --n_cri2 N_CRI2
  --gp_lam GP_LAM       weight of gradient penalty (WGAN-GP)
  --l1_lam L1_LAM       L1 loss of generator
  -g G                  GPU ID (negative value indicates CPU mode)
  --result_dir RESULT_DIR
  --neg_numbers [NEG_NUMBERS [NEG_NUMBERS ...]]
                        digits regarded as negative example
  --pos_numbers [POS_NUMBERS [POS_NUMBERS ...]]
                        digits regarded as positive example
```
