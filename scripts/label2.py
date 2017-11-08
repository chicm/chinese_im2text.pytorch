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


UNKNOWN_INDEX = -1
TRAIN_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/train/caption_train_images_20170902'
VAL_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/val/caption_validation_images_20170910'

from collections import Counter

def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']
    counter = Counter()
    for img in imgs:
        for caption in img['caption']:
             counter.update(caption) 
    word_counts = [x for x in counter.items() if x[1] >= count_thr]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, word_counts[:10])))
    print('total words:', len(word_counts))
    return word_counts

def encode_captions(imgs, params, wtoi):
    max_length = params['max_length']+2
    N = len(imgs)
    M = 0
    M = sum(len(img['caption']) for img in imgs)  # total number of captions
    print('Total number of images:' + str(N))
    print('Total number of captions:' + str(M))
    label_array = []
    info = []
    cap_ix = 0
    for i, img in enumerate(imgs):
        for caption in img['caption']:
            if len(caption) < 1:
                continue
            info_item = {}
            info_item['caption_id'] = cap_ix
            info_item['image_id'] = img['image_id']
            info_item['image_index'] = i
            info_item['split'] = img['split']
            info.append(info_item)
            L = np.zeros(max_length, dtype='uint32')
            L[0] = 1
            for k, w in enumerate(caption):
                L[k+1] = wtoi[w]
            L[len(caption)+1] = 2
            #L = [1] + [wtoi[w] for w in caption] + [2]
            label_array.append(L)
            cap_ix += 1
    json.dump(info, open(params['info_json'], 'w'), ensure_ascii=False, indent=4)
    return label_array

def process_labels(params):
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

    seed(123)  # make reproducible

    # create the vocab
    vocab = build_vocab(imgs, params)
    label_dict = ['<STR>', '<EOS>']
    label_dict.extend([w[0] for _, w in enumerate(vocab)])

    with open('data/dict.txt', 'w') as f:
        for i, w in enumerate(label_dict):
            f.write('{}\n'.format(w))

    itow = {i + 1: w for i, w in enumerate(label_dict)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(label_dict)}  # inverse table

    L = encode_captions(imgs, params, wtoi)

    # create output h5 file
    #N = len(imgs)
    f_lb = h5py.File(params['output_h5'], "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.close()


def check_labels(params):
    with open('data/dict.txt', 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    print(lines[:5])
    itow = {i+1 : w for i, w in enumerate(lines)}
    itow[0] = ''

    check_ix = [100, 567, 1234, 23456, 50000]
    train_imgs = json.load(open(params['input_train_json'], 'r'))
    info = json.load(open(params['info_json'], 'r'))

    h5_label_file = h5py.File(params['output_h5'], 'r', driver='core')
    L = h5_label_file['labels']
    print(h5_label_file['labels'].shape)
 
    for i in check_ix:
        info_item = info[i]
        for img in train_imgs:
            if img['image_id'] == info_item['image_id']:
                print('TRUE captions:')
                print(img['caption'])
                print('PROCESSED captions:')
                print([itow[ix] for ix in L[info_item['caption_id']]])

    h5_label_file.close()            
    pass

def main(params):
    #check_labels(params)
    process_labels(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='data/train/caption_train_annotations_20170902.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--input_val_json', default='data/val/caption_validation_annotations_20170910.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--num_val', default=30000, type=int,
                        help='number of images to assign to validation data (for CV etc)')
    parser.add_argument('--info_json', default='data/info.json', help='output json file')
    parser.add_argument('--output_h5', default='data/img_label.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=50, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='data',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_test', default=0, type=int,
                        help='number of test images (to withold until very very end)')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
