import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import os
import matplotlib.pyplot as plt
from cs285.infrastructure.env import *

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

import itertools
from itertools import permutations
# from sympy.utilities.iterables import multiset_permutations
import time
import multiprocessing

def do_stuff(perm, c=[]):
    # whatever.
    res = []
    e = Env(args)
    print('perm', perm)
    e.rreset(perm, res)
    c.append(None)
    res = np.array(res)
    data_path = './data/seq_perms/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(data_path + '{}.npy'.format(len(c)), res)


start = time.time()
# with multiprocessing.Pool(processes=4) as pool:
#     res = pool.map(do_stuff, permutations(list(range(256))))
# with multiprocessing.Pool(processes=4) as pool:
    # pool.map_async(do_stuff, permutations(list(range(256))))

for perm in permutations(list(range(256))):
    do_stuff(perm)

end = time.time()
print(end - start)
print('DONE')
