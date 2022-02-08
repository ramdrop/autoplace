# %%
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pickle
import tqdm
import json
from sklearn.neighbors import NearestNeighbors

from utils.common import load_para, load_split
from utils.common import Measure, Measure_Cosine
from utils.scancontext import ScanContext
from utils.rcshr import rcshr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default='se_te_dpr',
                    choices=['netvlad', 'netvlad_lstm', 'kidnapped', 'm2dp', 'utr', 'minkloc3d', 'scancontext', 'se', 'se_dpr', 'se_te', 'se_te_dpr'])
parser.add_argument('--rcshr', action='store_true', help='if defined/True, run RCSHR after evalutaion. Default: False')
parser.add_argument('--split', type=str, default='test', help='split to use', choices=['val', 'test'])

args = parser.parse_args()

# DUT:  'netvlad',' netvlad_lstm', 'kidnapped', 'm2dp', 'utr', 'minkloc3d', 'scancontext'
#       'encoder', 'encoder_dtr', 'encoder_lstm', 'encoder_dtr_lstm'
DUT = args.model

# RCSHR_ACTIVE: True, False
RCSHR_ACTIVE = args.rcshr

GT_REF = 'se_te_dpr'                                                           # make sure /pcl and /rcs exist in the dataset folder

RESULT_FOLDER = 'results'

if not os.path.exists(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)
print('==> Evaluating: {}...'.format(DUT))

with open('resume_path.json', 'r') as f:
    record = json.load(f)

if DUT in ['kidnapped', 'netvlad', 'netvlad_lstm', 'se', 'se_dpr', 'se_te', 'se_te_dpr', 'netvlad_front', 'deepseqslam']:
    resume_path = os.path.join('../..', record[DUT])
    gt, qFeat, dbFeat = load_para(resume_path)
    meas = Measure(gt=gt, qFeat=qFeat, dbFeat=dbFeat)
    recalls_at_n = meas.get_recall_at_n()
    recalls, precisions = meas.get_pr(vis=True)
    preds, dists = meas.get_preds()

    if RCSHR_ACTIVE and DUT in ['se', 'se_dpr', 'se_te', 'se_te_dpr']:
        info_db, info_val, structDir, imgDir = load_split(resume_path)
        print('structDir : ', structDir)
        print('imgDir : ', imgDir)
        print('info_db shape: ', info_db.shape)
        print('info_val shape: ', info_val.shape)
        preds, dists = meas.get_preds()

        if DUT == 'se':
            max_rcs, step_rcs, alpha = 0.04, 0.04, 0.07               # encoder_0722_2112.db        0.11%   0.27%   0.30%
        elif DUT == 'se_dpr':
            max_rcs, step_rcs, alpha = 0.06, 0.08, 0.33               # encoder_dtr_0809_1128.db    0.57%   0.70%   0.73%
        elif DUT == 'se_te':
            max_rcs, step_rcs, alpha = 0.32, 0.06, 0.34               # encoder_lstm_0730_1922.db   0.95%   0.38%   0.00%
        elif DUT == 'se_te_dpr':
            max_rcs, step_rcs, alpha = 0.02, 0.04, 0.41               # encoder_dtr_lstm_0722_2023.db   1.06%   0.76%   0.57%

        new_preds, new_dists = rcshr(preds, dists, max_rcs=max_rcs, step_rcs=step_rcs, alpha=alpha, info_db=info_db, info_val=info_val, rcs_path=os.path.join('../../', imgDir.replace('img', 'rcs')))
        meas = Measure(gt=gt, preds=new_preds, dists=new_dists)
        recalls_at_n_rcshr = meas.get_recall_at_n()
        recalls_rcshr, precisions_rcshr = meas.get_pr(vis=True)
        preds_rcshr, dists_rcshr = meas.get_preds()

        for index in [1, 5, 10]:
            print('{:.2f}%\t'.format(recalls_at_n_rcshr[index] - recalls_at_n[index]), end='')
        print('\n==> Recall@1/5/10: {:.1f}%, {:.1f}%, {:.1f}%'.format(recalls_at_n_rcshr[1], recalls_at_n_rcshr[5], recalls_at_n_rcshr[10]), end='')

        with open(os.path.join(RESULT_FOLDER, '{}_{}_results.pickle'.format(DUT, 'rcshr')), 'wb') as handle:
            feature = {'recalls_at_n': recalls_at_n_rcshr, 'recalls': recalls_rcshr, 'precisions': precisions_rcshr, 'preds': preds_rcshr, 'dists': dists_rcshr}
            pickle.dump(feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif DUT == 'm2dp':
    from m2dp import M2DP
    resume_path = os.path.join('..', record[GT_REF])
    info_db, info_val, structDir, _ = load_split(resume_path)
    gt, _, _ = load_para(resume_path)

    # get M2DP descriptors
    dbFeat = np.empty((info_db.shape[0], 192), dtype=np.float64)
    for index in tqdm.tqdm(range(len(info_db))):
        pcl_file = os.path.join('../../', structDir, 'pcl', '{:0>5d}.bin'.format(int(info_db[index, 0])))
        pcl = np.fromfile(pcl_file, dtype=np.float64).reshape(-1, 4)
        pcl = pcl[:, :3]
        des, A = M2DP(pcl)
        dbFeat[index, :] = des
    qFeat = np.empty((info_val.shape[0], 192), dtype=np.float64)
    for index in tqdm.tqdm(range(len(info_val))):
        pcl_file = os.path.join('../../', structDir, 'pcl', '{:0>5d}.bin'.format(int(info_val[index, 0])))
        pcl = np.fromfile(pcl_file, dtype=np.float64).reshape(-1, 4)
        pcl = pcl[:, :3]
        des, A = M2DP(pcl)
        qFeat[index, :] = des

    meas = Measure(gt=gt, qFeat=qFeat, dbFeat=dbFeat)
    recalls_at_n = meas.get_recall_at_n()
    recalls, precisions = meas.get_pr(vis=True)
    preds, dists = meas.get_preds()

elif DUT == 'minkloc3d':
    resume_path = os.path.join('..', record[GT_REF])
    gt, _, _ = load_para(resume_path)
    with open(os.path.join(RESULT_FOLDER, '{}_feature.pickle'.format(DUT)), 'rb') as f:
        feature = pickle.load(f)
        qFeat = feature['qFeat']
        dbFeat = feature['dbFeat']
    meas = Measure_Cosine(gt=gt, qFeat=qFeat, dbFeat=dbFeat)
    recalls_at_n = meas.get_recall_at_n()
    recalls, precisions = meas.get_pr(vis=True)
    preds, dists = meas.get_preds()

elif DUT == 'utr':
    resume_path = os.path.join('..', record[GT_REF])
    gt, _, _ = load_para(resume_path)
    with open(os.path.join(RESULT_FOLDER, '{}_feature.pickle'.format(DUT)), 'rb') as f:
        feature = pickle.load(f)
        qFeat = feature['qFeat']
        dbFeat = feature['dbFeat']
    meas = Measure_Cosine(gt, qFeat, dbFeat)
    recalls_at_n = meas.get_recall_at_n()
    recalls, precisions = meas.get_pr(vis=True)
    preds, dists = meas.get_preds()

elif DUT == 'scancontext':
    resume_path = os.path.join('..', record[GT_REF])
    info_db, info_val, structDir, _ = load_split(resume_path)
    gt, _, _ = load_para(resume_path)

    cache_file = os.path.join(RESULT_FOLDER, '{}_tmp_results.pickle'.format(DUT))
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            feature = pickle.load(f)
            new_dists = feature['new_dists']
            new_preds = feature['new_preds']
    else:
        dbFeat = []
        dbFeatKey = []
        for index in tqdm.tqdm(range(len(info_db)), ncols=50):
            sc_manager = ScanContext()
            pcl_file = os.path.join('../../', structDir, 'pcl', '{:0>5d}.bin'.format(int(info_db[index, 0])))
            # if index == 4842:
            # set_trace()
            sc = sc_manager.genSCs(pcl_file)
            sc_key = sc_manager.genKey(sc)
            dbFeat.append(sc)
            dbFeatKey.append(sc_key)

        qFeat = []
        qFeatKey = []
        for index in tqdm.tqdm(range(len(info_val)), ncols=50):
            sc_manager = ScanContext()
            pcl_file = os.path.join('../../', structDir, 'pcl', '{:0>5d}.bin'.format(int(info_val[index, 0])))
            sc = sc_manager.genSCs(pcl_file)
            sc_key = sc_manager.genKey(sc)
            qFeat.append(sc)
            qFeatKey.append(sc_key)

        knn = NearestNeighbors(n_jobs=1)
        knn.fit(dbFeatKey)
        dists, preds = knn.kneighbors(qFeatKey, 20)

        new_preds = []
        new_dists = []
        sc_manager = ScanContext()
        for q_ni in tqdm.tqdm(range(info_val.shape[0]), ncols=50):
            preds_q = preds[q_ni]
            sc_dists = []
            for pred_ni in preds_q:
                sc_dist = sc_manager.distance_sc(qFeat[q_ni], dbFeat[pred_ni])
                sc_dists.append(sc_dist)
            sc_dists = np.array(sc_dists, dtype=np.float64)
            new_ind = np.argsort(sc_dists)
            new_dists.append(sc_dists[new_ind])
            new_preds.append(preds_q[new_ind])

        with open(os.path.join(RESULT_FOLDER, '{}_tmp_results.pickle'.format(DUT)), 'wb') as handle:
            feature = {'new_dists': new_dists, 'new_preds': new_preds}
            pickle.dump(feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meas = Measure(gt=gt, preds=np.array(new_preds), dists=np.array(new_dists))
    recalls_at_n = meas.get_recall_at_n()
    recalls, precisions = meas.get_pr(vis=True)
    preds, dists = meas.get_preds()

RESULT_FILE = os.path.join(RESULT_FOLDER, '{}_results.pickle'.format(DUT))
with open(RESULT_FILE, 'wb') as handle:
    feature = {'recalls_at_n': recalls_at_n, 'recalls': recalls, 'precisions': precisions, 'preds': preds, 'dists': dists}
    pickle.dump(feature, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('\n==> Recall@1/5/10: {:.1f}%, {:.1f}%, {:.1f}%'.format(recalls_at_n[1], recalls_at_n[5], recalls_at_n[10]))
print('==> Precision-Recall saved.')
print('==> Done.')
