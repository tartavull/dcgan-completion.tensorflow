from glob import glob
from scipy.misc import imread, imsave
import re
from tqdm import tqdm
import numpy as np
import os.path

for img_path in tqdm(glob('/usr/people/it2/seungmount/research/datasets/lsum/bedroom_train_lmdb/raw/*.webp')):
  try:
    img = imread(img_path)
    folder, img_key = re.findall(r'lsum/(.*)/raw/(.*).webp',img_path)[0]
    save_path = '/usr/people/it2/seungmount/research/datasets/lsum_png/{}.png'.format(img_key)
    if os.path.exists(save_path):
      continue
    to_substract = np.array(img.shape) - np.array([256,256,3])
    left = np.floor(to_substract / 2.0).astype(int)
    right = np.ceil(to_substract / 2.0).astype(int) * -1
    right[right==0] = 256
    img = img[left[0]:right[0],left[1]:right[1],:]
    assert img.shape == (256,256,3)
    imsave(save_path, img)
  except Exception as e:
    print e
