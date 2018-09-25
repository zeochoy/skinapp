import torch
from torch import nn #, cuda, backends, FloatTensor, LongTensor, optim
#from torch.autograd import Variable
import torch.nn.functional as F
from model.resnext_50_32x4d import resnext_50_32x4d

from skinapp import app

resnext50 = resnext_50_32x4d()
model_meta = {resnext50:[8,6]}

arch = resnext50
c = 2
#is_multi = False
#is_reg = False

### -------- from fastai to get model
def split_by_idxs(seq, idxs):
    last = 0
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def cut_model(m, cut):
    #return list(m.children())[:cut] if cut else [m]
    return list(m)[:cut]

def children(m): return list(m)

def num_features(m):
    c=children(m)
    if len(c)==0: return None
    for l in reversed(c):
        if hasattr(l, 'num_features'): return l.num_features
        res = num_features(l)
        if res is not None: return res

#def resnext50(): return resnext_50_32x4d()

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

### -------- end of fastai

class SkinModel():
    """Class representing a convolutional network.
    Arguments:
        f: a model creation function (e.g. resnet34, vgg16, etc)
        c (int): size of the last layer
    """

    def __init__(self, f, c):
        self.f,self.c = f,c

        cut,_ = model_meta[f]
        layers = cut_model(f, cut)
        layers += [AdaptiveConcatPool2d(), Flatten()]
        #self.top_model = nn.Sequential(*layers)

        layers += [nn.BatchNorm1d(num_features=4096), nn.Dropout(p=0.25), nn.Linear(in_features=4096, out_features=512), nn.ReLU()]
        layers += [nn.BatchNorm1d(num_features=512), nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=self.c), nn.LogSoftmax()]
        self.model = nn.Sequential(*layers)

def get_skinmodel():
    m = SkinModel(arch, c)
    return m.model
