# %%
import os
import scipy.io as io
from sklearn.neighbors import NearestNeighbors
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random

# customized
import config as config
from utils.generic import Generic

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--struct_dir", type=str, default='/LOCAL/ramdrop/dataset/mmrec_dataset/7n5s_xy11')
parser.add_argument("--split", type=str, default='mini', choices=['trainval', 'test', 'mini'])
args = parser.parse_args()
paras = config.load_parameters(args.struct_dir)

random.seed(paras['SEED'])
np.random.seed(paras['SEED'])

print(paras['GROUP'])
record = config.load_parameters(args.struct_dir)

# %%
'''
1. get GT position, timestamps and find avaliable indices
'''
val = Generic('v1.0-trainval')

gt_pos_list_trainval = list()
timestamp_list_trainval = list()
safe_indices = list()
frame_index = 0
for scene_index in val.get_location_indices('boston-seaport'):
    val.to_scene(scene_index)
    nbr_samples = val.scene['nbr_samples']
    for j in range(nbr_samples):
        gt_pos = val.get_sample_abs_ego_pose()[:-1]
        gt_pos_list_trainval.append(gt_pos)
        timestamp = val.get_sample_data(channel='RADAR_FRONT')['timestamp']
        timestamp_list_trainval.append(timestamp)
        edge_frames_indices = set(range(0, paras['AVA_SEQ_LEN'] // 2, 1)) | set(range(nbr_samples - paras['AVA_SEQ_LEN'] // 2, nbr_samples, 1))
        if j not in edge_frames_indices:
            safe_indices.append(frame_index)
        frame_index += 1
        val.to_next_sample()

available_indices_trainval = set(safe_indices)
record['trainval: non-edge frames'] = len(safe_indices)
print('trainval non-edge / total frames:', frame_index, len(safe_indices))

val_test = Generic('v1.0-test')
gt_pos_list_test = list()
timestamp_list_test = list()
safe_indices = list()
frame_index = 0
for scene_index in val_test.get_location_indices('boston-seaport'):
    val_test.to_scene(scene_index)
    nbr_samples = val_test.scene['nbr_samples']
    for j in range(nbr_samples):
        gt_pos = val_test.get_sample_abs_ego_pose()[:-1]
        gt_pos_list_test.append(gt_pos)
        timestamp = val_test.get_sample_data(channel='RADAR_FRONT')['timestamp']
        timestamp_list_test.append(timestamp)
        edge_frames_indices = set(range(0, paras['AVA_SEQ_LEN'] // 2, 1)) | set(range(nbr_samples - paras['AVA_SEQ_LEN'] // 2, nbr_samples, 1))
        if j not in edge_frames_indices:
            safe_indices.append(frame_index)
        frame_index += 1
        val_test.to_next_sample()

available_indices_test = set(safe_indices)

record['test: non-edge frames'] = len(safe_indices)
print('test non-edge / total frames:', len(safe_indices))

gt_pos_list = gt_pos_list_trainval + gt_pos_list_test
timestamp_list = timestamp_list_trainval + timestamp_list_test
available_indices = list(available_indices_trainval)
for index in available_indices_test:
    available_indices.append(index + len(gt_pos_list_trainval))

assert len(available_indices) == len(available_indices_trainval) + len(available_indices_test), 'concatenate wrong'
print('total non-edge frames:', len(available_indices))

# %%
'''
3. seperate database+train_query from val_query+test_query based based on timestamps
'''
print('==> separate database, train, val, test..')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

gt_pos_list = np.array(gt_pos_list)
t_array = np.array(timestamp_list).reshape(-1, 1)
t_array = (t_array - min(t_array)) / (1e6 * 3600 * 24)  # unit: day

SEPERATE_TH = 105
# SEPERATE_TH = 103
fi_db_train, _ = np.where(t_array < SEPERATE_TH)
fi_val_test, _ = np.where(t_array >= SEPERATE_TH)

fi_db_train = set(fi_db_train) & set(available_indices)
fi_val_test = set(fi_val_test) & set(available_indices)

plt.rcParams['figure.figsize'] = (5, 5)
plt.plot(range(len(t_array)), t_array)
plt.hlines(SEPERATE_TH, 0, len(t_array), colors="r", linestyles="dashed")
plt.xlabel('frames index', fontsize=12)
plt.ylabel('relative day', fontsize=12)
plt.show()
plt.savefig(os.path.join(paras['STRUCT_DIR'], '3_timeline.jpg'))
plt.close()

fi_db_train = np.array(list(fi_db_train))
fi_val_test = np.array(list(fi_val_test))
plt.rcParams['figure.figsize'] = (5, 5)
plt.scatter(gt_pos_list[fi_db_train, 0], gt_pos_list[fi_db_train, 1], s=30, label='db+train')
plt.scatter(gt_pos_list[fi_val_test, 0], gt_pos_list[fi_val_test, 1], s=1, label='val+test')
plt.legend()
plt.savefig(os.path.join(paras['STRUCT_DIR'], '3_db+train_&_val+test.jpg'))
plt.close()

print('database + train_query:{}, val_query + test_query:{}'.format(fi_db_train.shape[0], fi_val_test.shape[0]))
print('database + train_query:{:.2f}, val_query + test_query:{:.2f}'.format(fi_db_train.shape[0] / len(available_indices), fi_val_test.shape[0] / len(available_indices)))

# %%
'''
4. generate database indices
'''
print('==> generate database..')

DIS_TH = 1  # Map Point Distance (m)
db_index_list = list()
query_index_list = list()

pos_whole = np.concatenate((np.arange(len(gt_pos_list), dtype=np.int32).reshape(-1, 1), np.array(gt_pos_list)), axis=1)
pos_db_train = pos_whole[fi_db_train]

pos_db = pos_db_train[0, :].reshape(1, -1)  # add the first frame
for i in range(1, pos_db_train.shape[0]):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(pos_db[:, 1:3])
    dis, index = knn.kneighbors(pos_db_train[i, 1:3].reshape(1, -1), 1, return_distance=True)

    if dis > DIS_TH:
        pos_db = np.concatenate((pos_db, pos_db_train[i, :].reshape(1, -1)), axis=0)

plt.rcParams['figure.figsize'] = (5, 5)
plt.scatter(pos_db_train[:, 1], pos_db_train[:, 2], s=30, label='db+train')
plt.scatter(pos_db[:, 1], pos_db[:, 2], s=1, label='db')
plt.legend()
plt.savefig(os.path.join(paras['STRUCT_DIR'], '4_train+db_&_db.jpg'))
plt.close()

print('database frames:{}'.format(pos_db.shape[0]))
# %%
'''
5. generate train_query, val_query, test_query indices
'''
print('==> generate train, val, test..')
fi_whole = pos_whole[:, 0].astype(int)  # fi, x, y

fi_db_train = pos_db_train[:, 0].astype(int)  # fi, x, y
fi_db = pos_db[:, 0].astype(int)  # fi, x, y

fi_train = list(set(fi_db_train) - set(fi_db))
# fi_train = np.random.choice(list(set(fi_db_train) - set(fi_db)), size=len(set(fi_db_train) - set(fi_db)), replace=False)
# fi_val = np.random.choice(fi_val_test, size=1500, replace=False)
# fi_test = np.random.choice(list(set(fi_val_test) - set(fi_val)), size=1500, replace=False)
fi_val = fi_val_test
fi_test = fi_val_test

fi_db = np.array(fi_db)
fi_train = np.array(fi_train)
fi_val = np.array(fi_val)
fi_test = np.array(fi_test)

pos_db = pos_whole[fi_db]  # fi, x, y
pos_train = pos_whole[fi_train]  # fi, x, y
pos_val = pos_whole[fi_val]  # fi, x, y
pos_test = pos_whole[fi_test]  # fi, x, y

print('db & whole:{}'.format(len(set(pos_db[:, 0]) & set(pos_whole[:, 0]))))
print('train & whole:{}'.format(len(set(pos_train[:, 0]) & set(pos_whole[:, 0]))))
print('val & whole:{}'.format(len(set(pos_val[:, 0]) & set(pos_whole[:, 0]))))
print('test & whole:{}'.format(len(set(pos_test[:, 0]) & set(pos_whole[:, 0]))))
print('train & val:{}'.format(len(set(pos_train[:, 0]) & set(pos_val[:, 0]))))
print('train & test:{}'.format(len(set(pos_train[:, 0]) & set(pos_test[:, 0]))))
print('val & test:{}'.format(len(set(pos_val[:, 0]) & set(pos_test[:, 0]))))

plt.rcParams['figure.figsize'] = (15, 5)
plt.subplot(1, 3, 1)
plt.scatter(pos_db[:, 1], pos_db[:, 2], s=30, label='db')
plt.scatter(pos_train[:, 1], pos_train[:, 2], s=1, label='train')
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(pos_db[:, 1], pos_db[:, 2], s=30, label='db')
plt.scatter(pos_val[:, 1], pos_val[:, 2], s=1, label='val')
plt.legend()
plt.subplot(1, 3, 3)
plt.scatter(pos_db[:, 1], pos_db[:, 2], s=30, label='db')
plt.scatter(pos_test[:, 1], pos_test[:, 2], s=1, label='test')
plt.legend()
plt.savefig(os.path.join(paras['STRUCT_DIR'], '4_db&train_val_test.jpg'))
plt.close()

# %%
'''
6.delete train/val/test queries who has no GT positive in the database
'''
print('==> refine train, val, test..')

DIS_TH = 9
knn = NearestNeighbors(n_neighbors=1)
knn.fit(pos_db[:, 1:3])

pos_train_new = list()
for i in range(len(pos_train)):
    dis, index = knn.kneighbors(pos_train[i, 1:3].reshape(1, -1), 1, return_distance=True)
    if dis < DIS_TH:
        pos_train_new.append(pos_train[i, :])
pos_train = np.array(pos_train_new)

pos_val_new = list()
for i in range(len(pos_val)):
    dis, index = knn.kneighbors(pos_val[i, 1:3].reshape(1, -1), 1, return_distance=True)
    if dis < DIS_TH:
        pos_val_new.append(pos_val[i, :])
pos_val = np.array(pos_val_new)

pos_test_new = list()
for i in range(len(pos_test)):
    dis, index = knn.kneighbors(pos_test[i, 1:3].reshape(1, -1), 1, return_distance=True)
    if dis < DIS_TH:
        pos_test_new.append(pos_test[i, :])
pos_test = np.array(pos_test_new)

print('db & whole:{}'.format(len(set(pos_db[:, 0]) & set(pos_whole[:, 0]))))
print('train & whole:{}'.format(len(set(pos_train[:, 0]) & set(pos_whole[:, 0]))))
print('val & whole:{}'.format(len(set(pos_val[:, 0]) & set(pos_whole[:, 0]))))
print('test & whole:{}'.format(len(set(pos_test[:, 0]) & set(pos_whole[:, 0]))))
print('train & val:{}'.format(len(set(pos_train[:, 0]) & set(pos_val[:, 0]))))
print('train & test:{}'.format(len(set(pos_train[:, 0]) & set(pos_test[:, 0]))))
print('val & test:{}'.format(len(set(pos_val[:, 0]) & set(pos_test[:, 0]))))

record['db frames'] = pos_db.shape[0]
record['train frames'] = pos_train.shape[0]
record['val frames'] = pos_val.shape[0]
record['test frames'] = pos_test.shape[0]

plt.rcParams['figure.figsize'] = (15, 5)
plt.subplot(1, 3, 1)
plt.scatter(pos_db[:, 1], pos_db[:, 2], s=30, label='db')
plt.scatter(pos_train[:, 1], pos_train[:, 2], s=1, label='train')
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(pos_db[:, 1], pos_db[:, 2], s=30, label='db')
plt.scatter(pos_val[:, 1], pos_val[:, 2], s=1, label='val')
plt.legend()
plt.subplot(1, 3, 3)
plt.scatter(pos_db[:, 1], pos_db[:, 2], s=30, label='db')
plt.scatter(pos_test[:, 1], pos_test[:, 2], s=1, label='test')
plt.legend()
plt.savefig(os.path.join(paras['STRUCT_DIR'], '6_db&train_val_test_refine.jpg'))
plt.close()

# %%
'''
7.generate .csv files
'''
print('==> generate databaset train, val, test indices in csv..')

for data, filename in zip([pos_db, pos_train, pos_val, pos_test], [['database'], ['train'], ['val'], ['test']]):
    dataframe = pd.DataFrame({'index': data[:, 0].astype(int), 'x': data[:, 1], 'y': data[:, 2]})
    csvfile = os.path.join(paras['STRUCT_DIR'], filename[0] + '.csv')
    dataframe.to_csv(csvfile, index=False, sep=',')
    print('{} saved'.format(csvfile))

# %%
'''
8.sanity check
'''
DEBUG = 0
error_flag = 0
if DEBUG:
    print('==> check all indices exist in pcl folder..')
    for data in [pos_db, pos_train, pos_val, pos_test]:
        for index in data[:, 0]:
            neighbor_indices = set(range(-(paras['AVA_SEQ_LEN'] // 2), paras['AVA_SEQ_LEN'] // 2 + 1, 1))
            for offset in neighbor_indices:
                if not os.path.exists(os.path.join(paras['PCL_DIR'], '{:0>5d}.bin'.format(int(index) + offset))):
                    print(index + offset)
                    error_flag = 1
    if error_flag:
        print('==> failed')
        exit(0)
    else:
        print('==> passed.')

# %%
'''
9.generate .mat for NetVLAD
'''
print('==> generate databaset train, val, test indices in .mat ..')

# whichSet = ['train', 'val']
whichSet = ['train', 'val', 'test']
dbImageFns = list()
for index in pos_db[:, 0]:
    dbImageFns.append('{:0>5d}.jpg'.format(int(index)))

utmDb = pos_db[:, 1:].T
numImages = pos_db.shape[0]

posDistThr = 18
posDistSqThr = posDistThr**2
nonTrivPosDistSqThr = 9**2

for ws in whichSet:
    # qImageFns = list()
    qImageFns = list()
    if ws == 'train':
        for i in range(len(pos_train[:, 0])):
            qImageFns.append('{:0>5d}.jpg'.format(int(pos_train[i, 0])))
        utmQ = pos_train[:, 1:].T
    elif ws == 'val':
        for i in range(len(pos_val[:, 0])):
            qImageFns.append('{:0>5d}.jpg'.format(int(pos_val[i, 0])))
        utmQ = pos_val[:, 1:].T
    elif ws == 'test':
        for i in range(len(pos_test[:, 0])):
            qImageFns.append('{:0>5d}.jpg'.format(int(pos_test[i, 0])))
        utmQ = pos_test[:, 1:].T

    numQueries = len(qImageFns)
    dbStruct = {
        'whichSet': ws,
        'dbImageFns': dbImageFns,
        'utmDb': utmDb,
        'qImageFns': qImageFns,
        'utmQ': utmQ,
        'numImages': numImages,
        'numQueries': numQueries,
        'posDistThr': posDistThr,
        'posDistSqThr': posDistSqThr,
        'nonTrivPosDistSqThr': nonTrivPosDistSqThr
    }

    matfile = os.path.join(paras['STRUCT_DIR'], 'nuscenes_{}.mat'.format(dbStruct['whichSet']))
    io.savemat(matfile, {'dbStruct': dbStruct})
    print('{} saved'.format(matfile))

config.update_parameters(args.struct_dir, record)
