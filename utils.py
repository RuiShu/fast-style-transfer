import scipy.misc
import numpy as np
import os
import sys
from glob import glob
import shutil

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def get_img(img_path, shape=None):
   img = scipy.misc.imread(img_path, mode='RGB')

   if shape:
       img = scipy.misc.imresize(img, shape)

   return img

def get_img_files(img_dir):
    f = glob(os.path.join(img_dir, '*.jpg'))
    f += glob(os.path.join(img_dir, '*.png'))
    return f

def delete_existing(path):
    if os.path.exists(path):
        print "Deleting existing {:s}".format(path)
        shutil.rmtree(path)
