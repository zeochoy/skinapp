import json, pdb, os, numpy as np, cv2, threading, math, random
#from urllib.request import urlopen

import torch
from torch import nn, cuda, backends, FloatTensor, LongTensor#, optim
from torch.autograd import Variable
import torch.nn.functional as F
#from torch.utils.model_zoo import load_url

from enum import IntEnum

from skinapp import app
from model.skinmodel import *

sz = 229

### -------- from fastai.dataset
def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to [0,1]
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


### -------- getting val_tfms to work without fastai import
class TfmType(IntEnum):
    """ Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            PIXEL: x and y are images and should be transformed in the same way.
                   Example: image segmentation.
            COORD: y are coordinates (i.e bounding boxes)
            CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
    """
    NO = 1
    PIXEL = 2
    COORD = 3
    CLASS = 4

class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3
    GOOGLENET = 4

class ChannelOrder():
    '''
    changes image array shape from (h, w, 3) to (3, h, w).
    tfm_y decides the transformation done to the y element.
    '''
    def __init__(self, tfm_y=TfmType.NO): self.tfm_y=tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 2)
        #if isinstance(y,np.ndarray) and (len(y.shape)==3):
        if self.tfm_y==TfmType.PIXEL: y = np.rollaxis(y, 2)
        elif self.tfm_y==TfmType.CLASS: y = y[...,0]
        return x,y

class Transforms():
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None: sz_y = sz
        self.sz,self.denorm,self.norm,self.sz_y = sz,denorm,normalizer,sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms + [crop_tfm, normalizer, ChannelOrder(tfm_y)]
    def __call__(self, im, y=None): return compose(im, y, self.tfms)
    def __repr__(self): return str(self.tfms)

def A(*a): return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

class Denormalize():
    """ De-normalizes an image, returning it to original format.
    """
    def __init__(self, m, s):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
    def __call__(self, x): return x*self.s+self.m


class Normalize():
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original image """
    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m=np.array(m, dtype=np.float32)
        self.s=np.array(s, dtype=np.float32)
        self.tfm_y=tfm_y

    def __call__(self, x, y=None):
        x = (x-self.m)/self.s
        if self.tfm_y==TfmType.PIXEL and y is not None: y = (y-self.m)/self.s
        return x,y

class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """
    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.store = threading.local()

    def set_state(self): pass
    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x,False)
        return (x, self.do_transform(y,True)) if y is not None else x

#     @abstractmethod
#     def do_transform(self, x, is_y): raise NotImplementedError

class CoordTransform(Transform):
    """ A coordinate transform.  """

    @staticmethod
    def make_square(y, x):
        r,c,*_ = x.shape
        y1 = np.zeros((r, c))
        y = y.astype(np.int)
        y1[y[0]:y[2], y[1]:y[3]] = 1.
        return y1

    def map_y(self, y0, x):
        y = CoordTransform.make_square(y0, x)
        y_tr = self.do_transform(y, True)
        return to_bb(y_tr, y)

    def transform_coord(self, x, ys):
        yp = partition(ys, 4)
        y2 = [self.map_y(y,x) for y in yp]
        x = self.do_transform(x, False)
        return x, np.concatenate(y2)

class Scale(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz: int
            target size to scale minimum size.
        tfm_y: TfmType
            type of y transformation.
    """
    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.sz_y = sz,sz_y

    def do_transform(self, x, is_y):
        if is_y: return scale_min(x, self.sz_y, cv2.INTER_NEAREST)
        else   : return scale_min(x, self.sz,   cv2.INTER_AREA   )

class RandomCrop(CoordTransform):
    """ A class that represents a Random Crop transformation.
    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        targ: int
            target size of the crop.
        tfm_y: TfmType
            type of y transformation.
    """
    def __init__(self, targ_sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.targ_sz,self.sz_y = targ_sz,sz_y

    def set_state(self):
        self.store.rand_r = random.uniform(0, 1)
        self.store.rand_c = random.uniform(0, 1)

    def do_transform(self, x, is_y):
        r,c,*_ = x.shape
        sz = self.sz_y if is_y else self.targ_sz
        start_r = np.floor(self.store.rand_r*(r-sz)).astype(int)
        start_c = np.floor(self.store.rand_c*(c-sz)).astype(int)
        return crop(x, start_r, start_c, sz)

class RandomRotate(CoordTransform):
    """ Rotates images and (optionally) target y.
    Rotating coordinates is treated differently for x and y on this
    transform.
     Arguments:
        deg (float): degree to rotate.
        p (float): probability of rotation
        mode: type of border
        tfm_y (TfmType): type of y transform
    """
    def __init__(self, deg, p=0.75, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.deg,self.p = deg,p
        if tfm_y == TfmType.COORD or tfm_y == TfmType.CLASS:
            self.modes = (mode,cv2.BORDER_CONSTANT)
        else:
            self.modes = (mode,mode)

    def set_state(self):
        self.store.rdeg = rand0(self.deg)
        self.store.rp = random.random()<self.p

    def do_transform(self, x, is_y):
        if self.store.rp: x = rotate_cv(x, self.store.rdeg,
                mode= self.modes[1] if is_y else self.modes[0],
                interpolation=cv2.INTER_NEAREST if is_y else cv2.INTER_AREA)
        return x

class RandomDihedral(CoordTransform):
    """
    Rotates images by random multiples of 90 degrees and/or reflection.
    Please reference D8(dihedral group of order eight), the group of all symmetries of the square.
    """
    def set_state(self):
        self.store.rot_times = random.randint(0,3)
        self.store.do_flip = random.random()<0.5

    def do_transform(self, x, is_y):
        x = np.rot90(x, self.store.rot_times)
        return np.fliplr(x).copy() if self.store.do_flip else x


def lighting(im, b, c):
    """ Adjust image balance and contrast """
    if b==0 and c==1: return im
    mu = np.average(im)
    return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x

class RandomScale(CoordTransform):
    """ Scales an image so that the min size is a random number between [sz, sz*max_zoom]
    This transforms (optionally) scales x,y at with the same parameters.
    Arguments:
        sz: int
            target size
        max_zoom: float
            float >= 1.0
        p : float
            a probability for doing the random sizing
        tfm_y: TfmType
            type of y transform
    """
    def __init__(self, sz, max_zoom, p=0.75, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz,self.max_zoom,self.p,self.sz_y = sz,max_zoom,p,sz_y

    def set_state(self):
        min_z = 1.
        max_z = self.max_zoom
        #if isinstance(self.max_zoom, collections.Iterable):
        #    min_z, max_z = self.max_zoom
        self.store.mult = random.uniform(min_z, max_z) if random.random()<self.p else 1
        self.store.new_sz = int(self.store.mult*self.sz)
        if self.sz_y is not None: self.store.new_sz_y = int(self.store.mult*self.sz_y)


    def do_transform(self, x, is_y):
        if is_y: return scale_min(x, self.store.new_sz_y, cv2.INTER_AREA if self.tfm_y == TfmType.PIXEL else cv2.INTER_NEAREST)
        else : return scale_min(x, self.store.new_sz, cv2.INTER_AREA )

transforms_basic    = [RandomRotate(10), RandomLighting(0.05, 0.05)]
#transforms_side_on  = transforms_basic + [RandomFlip()]
transforms_top_down = transforms_basic + [RandomDihedral()]

imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
stats = imagenet_stats

tfm_norm = Normalize(*stats, TfmType.NO)
tfm_denorm = Denormalize(*stats)

def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None, tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT):
    """
    Generate a standard set of transformations

    Arguments
    ---------
     normalizer :
         image normalizing function
     denorm :
         image denormalizing function
     sz :
         size, sz_y = sz if not specified.
     tfms :
         iterable collection of transformation functions
     max_zoom : float,
         maximum zoom
     pad : int,
         padding on top, left, right and bottom
     crop_type :
         crop type
     tfm_y :
         y axis specific transformations
     sz_y :
         y size, height
     pad_mode :
         cv2 padding style: repeat, reflect, etc.

    Returns
    -------
     type : ``Transforms``
         transformer for specified image operations.

    See Also
    --------
     Transforms: the transformer object returned by this function
    """
    if tfm_y is None: tfm_y=TfmType.NO
    if tfms is None: tfms=[]
    #elif not isinstance(tfms, collections.Iterable): tfms=[tfms]
    if sz_y is None: sz_y = sz
    scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None else Scale(sz, tfm_y, sz_y=sz_y)]
    if pad: scale.append(AddPadding(pad, mode=pad_mode))
    if crop_type!=CropType.GOOGLENET: tfms=scale+tfms
    return Transforms(sz, tfms, normalizer, denorm, crop_type,
                      tfm_y=tfm_y, sz_y=sz_y)

crop_fn_lu = {CropType.RANDOM: RandomCrop}

def compose(im, y, fns):
    """ apply a collection of transformation functions fns to images
    """
    for fn in fns:
        #pdb.set_trace()
        im, y =fn(im, y)
    return im if y is None else (im, y)

def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
    """
    r,c,*_ = im.shape
    ratio = targ/min(r,c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz, interpolation=interpolation)

def scale_to(x, ratio, targ):
    '''
    no clue, does not work.
    '''
    return max(math.floor(x*ratio), targ)

def crop(im, r, c, sz):
    '''
    crop image into a square of size sz,
    '''
    return im[r:r+sz, c:c+sz]

def rand0(s): return random.random()*(s*2)-s

def rotate_cv(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotate an image by deg degrees
    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

### -------- end val_tfms stuff

def to_np(v):
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if isinstance(v, torch.cuda.HalfTensor): v=v.float()
    return v.cpu().numpy()

### -------- tailor funcs
def preproc_img(img):
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=0, crop_type=CropType.RANDOM, tfm_y=None, sz_y=None, tfms=transforms_top_down, max_zoom=1.2)
    trans_img = val_tfm(img)
    return Variable(torch.FloatTensor(trans_img)).unsqueeze_(0)

def get_predictions(img):
    img_t = preproc_img(img)
    model  = load_model()

    #make predictions
    model.eval()
    res = model(img_t)
    probs = np.exp(to_np(res))
    preds = int(np.argmax(probs, axis=1))

    cat_dict = {0:'benign', 1:'malignant'}
    return {"cat":cat_dict[preds], "prob":np.amax(probs)*100}

def load_model():
    dst = app.config['MODEL_FILE']
    model = get_skinmodel()
    model.load_state_dict(torch.load(dst))
    #if os.path.isfile(dst):
    #    model = torch.load(dst)
    #else:
    #    dl_url = 'https://xxx'
    #    with urlopen(dl_url) as u, NamedTemporaryFile(delete=False) as f:
    #        f.write(u.read())
    #        shutil.move(f.name, dst)
    #    model = torch.load(dst)
    return model
