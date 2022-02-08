# %%
# public
from PIL import Image
import os
import tqdm
import numpy as np
from ipdb import set_trace
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import copy

# private
import preprocess.config as config
from preprocess.utils.generic import Generic
from preprocess.utils.pcl_operation import normalize_feature, rescale


# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--struct_dir", type=str, default='/LOCAL/ramdrop/dataset/mmrec_dataset/7n5s_xy11')
parser.add_argument("--split", type=str, default='trainval', choices=['trainval', 'test', 'mini'])
parser.add_argument("--img", type=int, default=1)
parser.add_argument("--pcl", type=int, default=1)
parser.add_argument("--rcs", type=int, default=1)
args = parser.parse_args()

paras = config.load_parameters(args.struct_dir)
print(paras['GROUP'])


# %% for debug
# paras = config.load_parameters('/LOCAL/ramdrop/dataset/mmrec_dataset/7n5s_xy11')
# print(paras['GROUP'])
# print(paras['IMG_DIR'])

val = Generic('v1.0-{}'.format(args.split))
boston_indices = val.get_location_indices('boston-seaport')

# %%

def pcl_to_img_PIL(pcl_input, frame_index, output_dir):
    img_matrix = np.zeros((2 * paras['MEASURE_RANGE'], 2 * paras['MEASURE_RANGE']), dtype='float')

    pcl_input[:, :2] = np.rint(pcl_input[:, :2]) + paras['MEASURE_RANGE']
    pcl_candidate = []
    for index, point in enumerate(pcl_input):
        if point[0] <= 2 * paras['MEASURE_RANGE'] - 1 and point[0] >= 0 and point[1] <= 2 * paras['MEASURE_RANGE'] - 1 and point[1] >= 0:
            pcl_candidate.append(pcl_input[index])

    pcl_candidate = np.array(pcl_candidate, dtype=np.float64)

    if paras['FEATURE'] == 'r':
        pcl_candidate = normalize_feature(pcl_input=pcl_candidate, feature_channel=[5], target='01')
        for row in pcl_candidate:
            img_matrix[int(row[0]), int(row[1])] = row[5]
    elif paras['FEATURE'] == 't':
        pcl_candidate = normalize_feature(pcl_input=pcl_candidate, feature_channel=[17], target='01')
        for row in pcl_candidate:
            img_matrix[int(row[0]), int(row[1])] = row[17]
    elif paras['FEATURE'] == '1':
        pcl_candidate = np.array(pcl_candidate, dtype=np.int32)
        img_matrix[pcl_candidate[:, 0], pcl_candidate[:, 1]] = 1

    img_matrix = np.expand_dims(img_matrix, axis=2)
    img_matrix = np.repeat(img_matrix, 3, axis=2)

    img = Image.fromarray(np.uint8(img_matrix * 255.0)).convert('RGB')
    img.save(os.path.join(output_dir, '{:0>5d}.jpg'.format(frame_index)))
    # img.save(os.path.join('tmp', '{:0>5d}.jpg'.format(frame_index)))
    pass

if args.split == 'trainval':
    frame_index = 0
elif args.split == 'test':
    frame_index = 18785
elif args.split == 'mini':
    frame_index = 90000

for scene_index in tqdm.tqdm(boston_indices):
    val.to_scene(scene_index)
    nbr_samples = val.scene['nbr_samples']
    for j in range(nbr_samples):
        # if frame_index < 2659:
        #     frame_index += 1
        #     val.to_next_sample()
        #     continue
        # set_trace()
        if paras['REMOVE']:
            pcl = val.get_pcl_pano_filtered([scene_index, j], chan='RADAR_FRONT', ref_chan='RADAR_FRONT', nsweeps=paras['NSWEEP'])
        else:
            pcl = val.get_pcl_pano(chan='RADAR_FRONT', ref_chan='RADAR_FRONT', nsweeps=paras['NSWEEP'])

        # ------------------------------------- generate imgs ------------------------------------ #
        if args.img:
            pcl_img = copy.deepcopy(pcl)
            pcl_to_img_PIL(pcl_input=pcl_img, frame_index=frame_index, output_dir=paras['IMG_DIR'])

        # ------------------------------------- generate rcs ------------------------------------- #
        if args.rcs:
            pcl_rcs = copy.deepcopy(pcl)
            pcl_candidate = []
            for index, point in enumerate(pcl_rcs):
                if point[0] <= paras['MEASURE_RANGE'] and point[0] >= -paras['MEASURE_RANGE'] and point[1] <= paras['MEASURE_RANGE'] and point[1] >= -paras['MEASURE_RANGE']:
                    pcl_candidate.append(pcl_rcs[index])

            pcl_candidate = np.array(pcl_candidate, dtype=np.float64)

            pcl_candidate = normalize_feature(pcl_input=pcl_candidate, feature_channel=[5], target='01')
            rcs = pcl_candidate[:, 5]
            rcs.tofile(os.path.join(paras['RCS_DIR'], '{:0>5d}.bin'.format(frame_index)))
            # rcs.tofile(os.path.join('tmp', '{:0>5d}.bin'.format(frame_index)))

        # ------------------------------------- generate pcl ------------------------------------- #
        if args.pcl:
            pcl_pcl = copy.deepcopy(pcl)
            pcl_normalized = normalize_feature(pcl_pcl, feature_channel=[5, 17])
            # pcl_normalized = pcl
            pcl_rescaled = rescale(pcl_normalized, measure_range=paras['MEASURE_RANGE'])
            pcl_aligned = pcl_rescaled
            if paras['FEATURE'] == 'r':
                pcl_aligned[:, 3] = pcl_aligned[:, 5]
            elif paras['FEATURE'] == 't':
                pcl_aligned[:, 3] = pcl_aligned[:, 17]
            elif paras['FEATURE'] == '1':
                pcl_aligned[:, 3] = np.ones((pcl_aligned.shape[0], ))
            else:
                print(paras['FEATURE'])
                print('undefined feature')
                exit(1)

            if paras['PSEUDO_Z'] == 0:
                pcl_aligned[:, 2] = np.zeros((pcl_aligned.shape[0], ))
            elif paras['PSEUDO_Z'] == 1:
                pcl_aligned[:, 2] = np.ones((pcl_aligned.shape[0], ))
            else:
                print(paras['PSEUDO_Z'])
                print('undefined pseudo z')
                exit(1)

            pcl_aligned[:, :4].tofile(os.path.join(paras['PCL_DIR'], '{:0>5d}.bin'.format(frame_index)))  # [:, 0,1,2,3]

        # ------------------------------------ to next sample ------------------------------------ #
        frame_index += 1
        val.to_next_sample()
        del pcl
