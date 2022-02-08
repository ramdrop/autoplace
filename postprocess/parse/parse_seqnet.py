# %%
import os

os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
import pickle
import tqdm
import json
import importlib
import torch
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from utils.common import Measure
from utils.common import load_para, load_split
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--phase",
    type=str,
    default='generate_features',
    choices=['generate_features', 'match'])
args = parser.parse_args()

# ['generate_features', 'match']
PHASE = args.phase

class flag_to_opt:
    def __init__(self, resume_path) -> None:
        self.flag = self.load_opt(resume_path)
        self.resume = resume_path
        self.img_dir = self.flag['imgDir']
        self.cGPU = self.flag['cGPU']
        self.num_clusters = self.flag['num_clusters']
        self.encoder_dim = self.flag['encoder_dim']
        self.output_dim = self.flag['output_dim']
        try:
            self.w = self.flag['w']
        except:
            print('Warning: No w in .json, but it is okay if you know what you are doing.')
        self.fromscratch = self.flag['fromscratch']
        self.net = self.flag['net']
        self.seqLen = self.flag['seqLen']
        self.structDir = self.flag['structDir']

    def load_opt(self, resume_path):
        model_parameter_file = os.path.join(resume_path, 'flags.json')
        assert os.path.exists(model_parameter_file), 'model parameter file does not exists, you may have broken your log directory'
        with open(model_parameter_file, 'r') as f:
            flag = json.load(f)
        return flag


def get_model(resume_path):
    opt = flag_to_opt(resume_path)
    torch.cuda.set_device(1)
    device = torch.device("cuda")
    print('device: {} {}'.format(device, torch.cuda.current_device()))
    reparsed_network = '{}.{}.networks.{}'.format(opt.resume.split('/')[-2], opt.resume.split('/')[-1], opt.net)
    network = importlib.import_module(reparsed_network)
    model = network.get_model(opt, require_init=False)
    resume_ckpt = os.path.join(opt.resume, 'checkpoint_best.pth.tar')
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model, device

with open('resume_path.json', 'r') as f:
    record = json.load(f)
NetVLAD_RESUME = record['netvlad']
S3SeqNet_RESUME = record['s3_seqnet']
S1SeqNet_RESUME = record['s1_seqnet']

assert os.path.exists(NetVLAD_RESUME), 'log not exist, check `netvlad` resume path in resume_path.json'
opt = flag_to_opt(NetVLAD_RESUME)
gt, qFeat, dbFeat = load_para(NetVLAD_RESUME)
info_db, info_val, structDir, imgDir = load_split(NetVLAD_RESUME)

# Dataset path
IMG_DIR = os.path.join('../..', imgDir)
FEAT_DIR3 = os.path.join('../../', structDir, 'seqnet_feat3')
FEAT_DIR1 = os.path.join('../../', structDir, 'seqnet_feat1')

if not os.path.exists(FEAT_DIR1):
    os.mkdir(FEAT_DIR1)
if not os.path.exists(FEAT_DIR3):
    os.mkdir(FEAT_DIR3)

# sanity check
meas = Measure(gt=gt, qFeat=qFeat, dbFeat=dbFeat)
recalls_at_n = meas.get_recall_at_n()
print('NetVLAD:', recalls_at_n)

if PHASE == 'generate_features':
    # ----------- A. S3 SeqNet: generate all features and convert them from 32768D to 4096D ----------- #
    # the generated features are used to train S3 SeqNet
    model, device = get_model(NetVLAD_RESUME)

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    with torch.no_grad():
        imgs_list = os.listdir(IMG_DIR)
        imgs_list.sort(key=lambda x: int(x[:5]))
        allFeat = np.empty((len(imgs_list), 9216))
        for index, img in enumerate(tqdm.tqdm(imgs_list, ncols=40)):
            img = Image.open(os.path.join(IMG_DIR, img))
            img = input_transform(img).to(device)
            img = torch.unsqueeze(img, 0)
            img = torch.unsqueeze(img, 0)
            feature = model(img)
            allFeat[index, :] = feature.detach().cpu().numpy()
            del img, feature

    q_index = np.array([x[0] for x in info_val], dtype=np.int32)
    db_index = np.array([x[0] for x in info_db], dtype=np.int32)
    qFeat = allFeat[q_index]
    dbFeat = allFeat[db_index]
    print('fit PCA..')
    pca = PCA(n_components=4096, svd_solver='auto')
    pca.fit(dbFeat)
    qFeat = pca.transform(qFeat)
    dbFeat = pca.transform(dbFeat)

    # sanity check: features are the same as the original network's output
    meas = Measure(gt=gt, qFeat=qFeat, dbFeat=dbFeat)
    recalls_at_n = meas.get_recall_at_n()
    print('NetVLAD PCA Features:', recalls_at_n)

    pca_allFeat = pca.transform(allFeat)
    for index, feature in enumerate(tqdm.tqdm(pca_allFeat, ncols=40)):
        feature.tofile(os.path.join(FEAT_DIR3, '{:0>5d}.bin'.format(index)))
    print('extracted and saved all NetVLAD PCA features in *.bin')
    # sanity check: stored features are in the right dimension
    feature = np.fromfile(os.path.join(FEAT_DIR3, '{:0>5d}.bin'.format(0)), dtype=np.float64).reshape(-1, 1)
    assert feature.shape[0] == 4096, 'Error: Feature dimension is not as expected.'


elif PHASE == 'match':
    # ------------------------------ B. S1 SeqNet: generate all features ------------------------------ #
    assert os.path.exists(S3SeqNet_RESUME), 'log not exist, check `s3_seqnet` resume path in resume_path.json'
    assert os.path.exists(S1SeqNet_RESUME), 'log not exist, check `s1_seqnet` resume path in resume_path.json'

    model, device = get_model(S1SeqNet_RESUME)

    model.eval()
    with torch.no_grad():
        feats_list = os.listdir(FEAT_DIR3)
        feats_list.sort(key=lambda x: int(x[:5]))
        allFeat = np.empty((len(feats_list), 4096))
        for index, feat in enumerate(tqdm.tqdm(feats_list, ncols=40)):
            feat = np.fromfile(os.path.join(FEAT_DIR3, feat), dtype=np.float64).reshape(-1, 1)
            feat = feat.astype(np.float32)
            feat = torch.from_numpy(feat).to(device)
            feat = torch.unsqueeze(feat, 0)
            feat = torch.unsqueeze(feat, 0)
            feat_trans = model(feat)
            allFeat[index, :] = feat_trans.detach().cpu().numpy()
            del feat, feat_trans

    for index, feature in enumerate(tqdm.tqdm(allFeat, ncols=40)):
        feature.tofile(os.path.join(FEAT_DIR1, '{:0>5d}.bin'.format(index)))
    print('extract and save all S1 SeqNet features in *.bin')

    # %%
    # ------------------------------- C. S3-SeqNetFeature/S1-SeqNetFeature + SeqMatcher ------------------------------- #
    def get_seq3_predictions(num_k):
        gt, qFeat, dbFeat = load_para(S3SeqNet_RESUME)
        knn = NearestNeighbors(n_jobs=1)
        knn.fit(dbFeat)
        dists, predictions = knn.kneighbors(qFeat, num_k)
        return gt, predictions

    def get_seq1_predictions(num_k):
        gt, qFeat, dbFeat = load_para(S1SeqNet_RESUME)
        knn = NearestNeighbors(n_jobs=1)
        knn.fit(dbFeat)
        dists, predictions = knn.kneighbors(qFeat, num_k)
        return gt, predictions

    def get_seq1_feat_dist(q_index_fi, db_index_fi):
        qfeat = np.fromfile(os.path.join(FEAT_DIR1, '{:0>5d}.bin'.format(int(q_index_fi))), dtype=np.float64).reshape(-1, 1)
        dbfeat = np.fromfile(os.path.join(FEAT_DIR1, '{:0>5d}.bin'.format(int(db_index_fi))), dtype=np.float64).reshape(-1, 1)
        dist = LA.norm(qfeat - dbfeat)
        return dist

    gt, seq3_preds = get_seq3_predictions(num_k=100)
    meas = Measure(gt=gt, preds=seq3_preds)
    print('S3-SeqNet:', meas.get_recall_at_n())
    gt, seq1_preds = get_seq1_predictions(num_k=100)
    meas = Measure(gt=gt, preds=seq1_preds)
    print('S1-SeqNet:', meas.get_recall_at_n())
    new_preds = []
    new_dists = []
    for q_ni in tqdm.tqdm(range(info_val.shape[0]), ncols=40):
        q_fi = info_val[q_ni][0]
        seq3_preds_cur = seq3_preds[q_ni]                                            # predictions by S3-SeqNet
        match_scores = []
        for pred_ni in seq3_preds_cur:
            pred_fi_center = info_db[pred_ni][0]
            match_score = 0
            for offset in [-1, 0, 1]:
                pred_fi = pred_fi_center + offset
                feat_dist = get_seq1_feat_dist(q_fi, pred_fi)
                match_score += feat_dist
            match_score = match_score
            match_scores.append(match_score)
        match_scores = np.array(match_scores, dtype=np.float64)
        new_ind = np.argsort(match_scores)
        new_preds.append(seq3_preds_cur[new_ind])
        new_dists.append(match_scores[new_ind])

    # -------------------------------------- Save SeqNet results ------------------------------------- #
    meas = Measure(gt=gt, preds=np.array(new_preds), dists=np.array(new_dists))
    recalls_at_n = meas.get_recall_at_n()
    recalls, precisions = meas.get_pr(vis=True)
    preds, dists = meas.get_preds()

    print('S3-SeqNetFeature/S1-SeqNetFeature + SeqMatcher:', recalls_at_n)

    if not os.path.exists('results'):
        os.mkdir('results')
    with open(os.path.join('results/seqnet_results.pickle'), 'wb') as handle:
        feature = {'recalls_at_n': recalls_at_n, 'recalls': recalls, 'precisions': precisions, 'preds': preds, 'dists': dists}
        pickle.dump(feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
