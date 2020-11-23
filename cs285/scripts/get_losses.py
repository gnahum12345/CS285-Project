
import itertools
import os
from cs285.infrastructure.env import *

def brute(items, n):
    """generate tuples of len 1 to n with items chosen in a given sequence"""
    items = list(items)
    for k in range(1, n+1):
        for t in itertools.product(items, repeat = k):
            yield t


for t in brute(range(0, 256), 256):
    print(t)

path = '/mikRAID/frank/data/cube_knees/train_ksp_slices'
args = {}
args['mask_shape'] = (256,256)
args['ksp_data_path']  = path
args['coord'] = None
args['total_var'] = False # use total variation or not.
args['loss_type'] = 1 #  this is 1 for l1, or 2 for l2.
args['cartesian'] = True  # true or false.
args['history_length'] = 10000

e = Env(args)
