import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MyFullModel(nn.Module):
    def __init__(self, cnn_model, decoder_model):
        super(MyFullModel, self).__init__()
        self.cnn_model = cnn_model
        self.decoder_model = decoder_model

    def forward(img, label):
        fc, att = self.cnn_model(img)
        preds = self.decoder_model(fc,att,lable)
        return preds
