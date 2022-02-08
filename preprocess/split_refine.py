# %%
import scipy.io as io
import pandas as pd
import os
from os.path import join
import numpy as np
import config as config
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--struct_dir", type=str, default='/LOCAL/ramdrop/dataset/mmrec_dataset/7n5s_xy11')
args = parser.parse_args()
paras = config.load_parameters(args.struct_dir)

random.seed(paras['SEED'])
np.random.seed(paras['SEED'])

# where to find structure files
test_mat_file = join(args.struct_dir, 'nuscenes_test.mat')
# test_mat_file = '/LOCAL/ramdrop/dataset/tmp/7n5s_xy11/nuscenes_test.mat'

mat = io.loadmat(test_mat_file)
matStruct = mat['dbStruct'].item()

whichSet = matStruct[0].item()

dbImage = matStruct[1]
utmDb = matStruct[2].T
numDb = matStruct[5].item()

qImage = matStruct[3]
utmQ = matStruct[4].T
numQ = matStruct[6].item()

posDistThr = matStruct[7].item()
posDistSqThr = matStruct[8].item()
nonTrivPosDistSqThr = matStruct[9].item()
# %%

val_ind = np.random.choice(numQ, size=int(numQ * 0.2), replace=False)
test_ind = set(range(numQ)) - set(val_ind)
assert len(set(val_ind) | set(test_ind)) == numQ, 'split error'
test_ind = np.array(list(test_ind))

qImage_val = qImage[val_ind]
utmQ_val = utmQ[val_ind]
numQ_val = len(val_ind)

qImage_test = qImage[test_ind]
utmQ_test = utmQ[test_ind]
numQ_test = len(test_ind)
print('val_ind[0]:', val_ind[0])               # 3360
print('test_ind[0]:', test_ind[0])               # 0
print('val length:', numQ_val)
print('test length:', numQ_test)

# %%
# dbIndices = [x[:5] for x in dbImage]
# dataframe = pd.DataFrame({'index': dbIndices, 'x': utmDb[:, 0], 'y': utmDb[:, 1]})
# dataframe.to_csv('tmp/database.csv', index=False, sep=',')

qIndices = [x[:5] for x in qImage_val]
dataframe = pd.DataFrame({'index': qIndices, 'x': utmQ_val[:, 0], 'y': utmQ_val[:, 1]})
dataframe.to_csv(join(args.struct_dir, 'val.csv'), index=False, sep=',')

qIndices = [x[:5] for x in qImage_test]
dataframe = pd.DataFrame({'index': qIndices, 'x': utmQ_test[:, 0], 'y': utmQ_test[:, 1]})
dataframe.to_csv(join(args.struct_dir, 'test.csv'), index=False, sep=',')

# %%
dbStruct = {
    'whichSet': 'val',
    'dbImageFns': dbImage,
    'utmDb': utmDb.T,
    'qImageFns': qImage_val,
    'utmQ': utmQ_val.T,
    'numImages': numDb,
    'numQueries': numQ_val,
    'posDistThr': posDistThr,
    'posDistSqThr': posDistSqThr,
    'nonTrivPosDistSqThr': nonTrivPosDistSqThr
}

matfile = join(args.struct_dir, 'nuscenes_val.mat')
io.savemat(matfile, {'dbStruct': dbStruct})

dbStruct = {
    'whichSet': 'test',
    'dbImageFns': dbImage,
    'utmDb': utmDb.T,
    'qImageFns': qImage_test,
    'utmQ': utmQ_test.T,
    'numImages': numDb,
    'numQueries': numQ_test,
    'posDistThr': posDistThr,
    'posDistSqThr': posDistSqThr,
    'nonTrivPosDistSqThr': nonTrivPosDistSqThr
}

matfile = join(args.struct_dir, 'nuscenes_test.mat')
io.savemat(matfile, {'dbStruct': dbStruct})

