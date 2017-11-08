from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, models, transforms
import os, json, random, glob
import numpy as np

TRAIN_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/train/480'
VAL_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/val/480'
TEST_IMG_DIR = '/home/chicm/ml/kgdata/ai/caption/test1/480'
MAX_LABEL_LENGTH = 50

with open('data/word_counts.txt', 'r') as f:
    vocab = f.readlines()
    vocab = [line.strip().split(',')[0] for line in vocab]

infos = json.load(open('data/ai_challenger.json', 'r'))
itow = infos['ix_to_word']

wtoi = {v: int(k) for k, v in itow.items()}
print(len(itow))
print(len(wtoi))
print([(w, i) for w, i in wtoi.items()][:10])

def pil_load(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ImgCaptionDataset(data.Dataset):
    def __init__(self, json_path, img_root, random_caption=True):
        self.json_path = json_path
        if json_path is None:
            self.img_ids = glob.glob(img_root+'/*.jpg')
        else:
            self.imgs = json.load(open(json_path, 'r'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.random_caption = random_caption
        self.img_root = img_root

    def __getitem__(self, index):
        if self.json_path is None:
            img_path = os.path.join(self.img_root, self.img_ids[index])
            #print(img_path, self.img_root, self.img_ids[index])
        else:
            img_path = os.path.join(self.img_root, self.imgs[index]['image_id'])
        img_data = self.transform(pil_load(img_path))
        
        if not self.json_path is None:
            ix = 0
            if self.random_caption:
                ix = random.randint(0, len(self.imgs[index]['caption'])-1)
            sentence = self.imgs[index]['caption'][ix]
            label = np.zeros(MAX_LABEL_LENGTH+2, dtype=np.long)
            mask = np.zeros(MAX_LABEL_LENGTH+2, dtype=np.float32)

            for i, w in enumerate(sentence.strip()):
                try:
                    label[i+1] = wtoi[w]
                except KeyError:
                    print('KEYERROR:{}'.format(w))
                mask[i+1] = 1
            #print(label)
            return img_data, label, mask
        else:
            #print(img['image_id'])
            return img_data, self.img_ids[index].split('/')[-1]

    def __len__(self):
        if self.json_path is None:
            return len(self.img_ids)
        else:
            return len(self.imgs)


print(len(itow))

def get_img_loader(json_path, img_root, batch_size=8):
    dset = ImgCaptionDataset(json_path, img_root)
    dloader = torch.utils.data.DataLoader(dset,  batch_size=batch_size, shuffle=False, num_workers=8)
    dloader.itow = itow
    dloader.vocab_size = len(itow)
    dloader.len = len(dset)
    dloader.seq_length = 50
    return dloader

def test_val_loader():
    loader = get_img_loader('/home/chicm/ml/kgdata/ai/caption/val/caption_validation_annotations_20170910.json', VAL_IMG_DIR)
    for i, data in enumerate(loader):
        pic,label, mask = data
        print(pic.size())
        print(label.size())
        print(mask.size())
        if i > 10:
            break

def test_test1_loader():
    loader = get_img_loader(None, TEST_IMG_DIR)
    print(len(loader))
    for i, data in enumerate(loader):
        pic, img_id = data
        print(pic.size())
        #print(img_id)
        #print(label.size())
        #print(mask.size())
        if i > 10:
            break


#test_val_loader()
#test_test1_loader()
