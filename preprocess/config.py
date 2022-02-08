import os
import json
# import shutil
import argparse

def export_parameters(args):
    # point cloud & img parameters
    NSWEEP = args.n_sweep                                                      # [(1,2,3,..)]<> frames - small time windows: how many frames to do overlap
    AVA_SEQ_LEN = args.ava_seq_len                                             # [(1,3,5,7,..)] <> frames   large time windows: [(1,3,5,7,..)]
    REMOVE = args.remove                                                       # [True, False] <> frames > True: remove moving objects, False, no action.
    FEATURE = args.feature                                                     # ['1':1, 'r':rcs, 't':timestamp]
    PSEUDO_Z = args.pseudo_z                                                   # [0, 1]
    MEASURE_RANGE = args.measure_range
    # random seed
    SEED = 12345

    GROUP = '{}n{}s_xy{}{}{}'.format(NSWEEP, AVA_SEQ_LEN, PSEUDO_Z, FEATURE, '_remove' if REMOVE else '')
    STRUCT_DIR = os.path.join(args.dataset_dir, GROUP)
    PCL_DIR = os.path.join(STRUCT_DIR, 'pcl')
    IMG_DIR = os.path.join(STRUCT_DIR, 'img')
    RCS_DIR = os.path.join(STRUCT_DIR, 'rcs')

    if os.path.exists(STRUCT_DIR):
        print('Warning: directory {} exists'.format(STRUCT_DIR))
    else:
        if not os.path.exists(args.dataset_dir):
            os.mkdir(args.dataset_dir)
        os.mkdir(STRUCT_DIR)
        os.mkdir(PCL_DIR)
        os.mkdir(IMG_DIR)
        os.mkdir(RCS_DIR)

    record = {}
    record['GROUP'] = GROUP
    record['NSWEEP'] = NSWEEP
    record['AVA_SEQ_LEN'] = AVA_SEQ_LEN
    record['REMOVE'] = REMOVE
    record['FEATURE'] = FEATURE
    record['PSEUDO_Z'] = PSEUDO_Z
    record['MEASURE_RANGE'] = MEASURE_RANGE
    record['SEED'] = SEED
    record['STRUCT_DIR'] = STRUCT_DIR
    record['PCL_DIR'] = PCL_DIR
    record['IMG_DIR'] = IMG_DIR
    record['RCS_DIR'] = RCS_DIR
    with open(os.path.join(STRUCT_DIR, 'pcl_parameter.json'), 'w') as f:
        json.dump(record, f, indent='')


def load_parameters(struct_dir):
    with open(os.path.join(struct_dir, 'pcl_parameter.json'), 'r') as f:
        record = json.load(f)
    return record


def update_parameters(struct_dir, update_record):
    record = load_parameters(struct_dir)
    record.update(update_record)
    with open(os.path.join(struct_dir, 'pcl_parameter.json'), 'w') as f:
        json.dump(record, f, indent='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sweep", type=int, default=7)
    parser.add_argument("--ava_seq_len", type=int, default=5,)
    parser.add_argument("--remove", action="store_true", default=0,)
    parser.add_argument("--feature", type=str, default='1', choices=['r', 't', '1'])
    parser.add_argument("--pseudo_z", type=int, default=1)
    parser.add_argument("--measure_range", type=int, default=100)
    parser.add_argument("--comment", type=str, default='')
    parser.add_argument("--dataset_dir", type=str, default='/LOCAL/ramdrop/dataset/mmrec_dataset')
    args = parser.parse_args()

    export_parameters(args)
