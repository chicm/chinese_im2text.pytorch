from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os

import opts
import models
import eval_utils
import argparse
import misc.utils as utils
import torch
from torch.autograd import Variable

from misc.resnet_utils import myResnet
import misc.resnet as resnet
from capdataset import get_img_loader

TEST_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/test1/480'
VAL_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/val/480'

opt = opts.parse_opt()

#opt.img_root = TEST_IMG_DIR
opt.img_root = VAL_IMG_DIR
#opt.output_file = 'test_out.json'
opt.output_file = 'results/val_1108.json'
opt.model = 'checkpoints/sat480/model.pth'

img_loader = get_img_loader(None, opt.img_root, batch_size=4)
opt.vocab_size = img_loader.vocab_size
opt.seq_length = 52

resnet = resnet.resnet152(pretrained=True)
my_resnet = myResnet(resnet)
my_resnet.cuda()
my_resnet.eval()

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

results = []
N = img_loader.len
n = 0

for i, data in enumerate(img_loader):
    img, image_id = data
    inputs = Variable(img.cuda())
    #print(inputs.size())
    fc, att = my_resnet(inputs)
    #print(fc.size(), att.size())

    seq, _ = model.sample_beam(fc, att, vars(opt))
    sents = utils.decode_sequence(img_loader.itow, seq)
    print('{}:{}'.format(image_id, sents))
    n += len(image_id)
    print('{} / {}'.format(n, N))

    for j, img_id in enumerate(image_id):
        result = {}
        result['image_id'] = img_id.split('.')[0]
        result['caption'] = sents[j]
        results.append(result)
    
    #if i>10:
    #    break

json.dump(results, open(opt.output_file, 'w'), ensure_ascii=False, indent=4)
