"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms

from misc.resnet_utils import myResnet
import jieba

TRAIN_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/train/224'
VAL_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/val/224'


from collections import Counter

def pil_load(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ImgDataset(data.Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        img = self.imgs[index]
        if img['split'] == 'train':
            img_path = TRAIN_IMG_DIR + '/' + img['image_id']
        else:
            img_path = VAL_IMG_DIR + '/' + img['image_id']
        
        img = pil_load(img_path)
        return self.transform(img)

    def __len__(self):
        return len(self.imgs)

def get_img_loader(imgs):
    dset = ImgDataset(imgs)
    dloader = torch.utils.data.DataLoader(dset,  batch_size=48, shuffle=False, num_workers=8)
    return dloader

def main(params):
    train_imgs = json.load(open(params['input_train_json'], 'r'))
    for i, img in enumerate(train_imgs):
        img['split'] = 'train'
    print(len(train_imgs))
    val_imgs = json.load(open(params['input_val_json'], 'r'))
    for i, img in enumerate(val_imgs):
        img['split'] = 'val'
    print(len(val_imgs))

    imgs = train_imgs
    imgs.extend(val_imgs)
    print(len(imgs))

    #imgs = imgs['images']

    seed(123)  # make reproducible
    #shuffle(imgs)  # shuffle the order
    #prepro_captions(imgs)



    import misc.resnet as resnet
    resnet_type = 'resnet152'
    if resnet_type == 'resnet101':
        resnet = resnet.resnet101()
        resnet.load_state_dict(torch.load('resnet/resnet101.pth'))
    else:
        resnet = resnet.resnet152(pretrained=True)
        #resnet.load_state_dict(torch.load('resnet/resnet152.pth'))
    my_resnet = myResnet(resnet)
    my_resnet.cuda()
    my_resnet.eval()

    # create output h5 file
    N = len(imgs)
    f_fc = h5py.File(params['output_h5'] + '_fc.h5', "w")
    f_att = h5py.File(params['output_h5'] + '_att.h5', "w")

    dset_fc = f_fc.create_dataset("fc", (N, 2048), dtype='float32')
    dset_att = f_att.create_dataset("att", (N, 14, 14, 2048), dtype='float32')
    fc_idx = 0
    att_idx = 0
    img_loader = get_img_loader(imgs)
    for i, data in enumerate(img_loader):
        inputs = Variable(data.cuda())
        #print(data.size())
        tmp_fc, tmp_att = my_resnet(inputs)
        tmp_fc = tmp_fc.data.cpu().float().numpy()
        tmp_att = tmp_att.data.cpu().float().numpy()
        #print(tmp_fc.shape)
        #print(tmp_att.shape)
        for j in range(len(tmp_fc)):
            dset_fc[fc_idx] = tmp_fc[j]
            fc_idx += 1
        for j in range(len(tmp_att)):
            dset_att[att_idx] = tmp_att[j]
            att_idx += 1;
        if i % 100 == 0:
            print('processing %d/%d (%.2f%% done)' % (fc_idx, N, fc_idx * 100.0 / N))

    f_fc.close()
    f_att.close()
    exit()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='data/train/caption_train_annotations_20170902.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--input_val_json', default='data/val/caption_validation_annotations_20170910.json',
                        help='input json file to process into hdf5')
    #parser.add_argument('--num_val', default=30000, type=int,
    #                    help='number of images to assign to validation data (for CV etc)')
    #parser.add_argument('--output_json', default='data/ai_challenger.json', help='output json file')
    parser.add_argument('--output_h5', default='data/res152_224', help='output h5 file')

    # options
    #parser.add_argument('--max_length', default=50, type=int,
    #                    help='max length of a caption, in number of words. captions longer than this get clipped.')
    #parser.add_argument('--images_root', default='data',
    #                    help='root location in which images are stored, to be prepended to file_path in input json')
    #parser.add_argument('--word_count_threshold', default=1, type=int,
    #                    help='only words that occur more than this number of times will be put in vocab')
    #parser.add_argument('--num_test', default=0, type=int,
    #                    help='number of test images (to withold until very very end)')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
