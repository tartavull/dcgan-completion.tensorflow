import h5py
import numpy as np
from scipy.misc import imsave
import scipy.signal
from glob import glob
from tqdm import tqdm
import skimage.measure

i = 0
for path in tqdm(glob('/usr/people/it2/seungmount/research/datasets/piriform_157x2128x2128/train/completions2/threshold_0.*/*.h5')):
  with h5py.File(path) as f:

    arr = f['output'][:,:,:]
    # ds = scipy.signal.decimate(arr,q=5,axis=1)
    # ds = scipy.signal.decimate(ds,q=5,axis=2)

    ds = skimage.measure.block_reduce(arr,(1,5,5))
    reshaped = ds.reshape(32*32,32).astype(np.float32)
    imsave('/usr/people/it2/seungmount/research/datasets/neuron_png/{:06d}.png'.format(i), reshaped)
    i += 1