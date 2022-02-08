import os
import pickle
from ipdb.__main__ import set_trace
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tqdm
import faiss
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_para(resume_path, split='test'):
    with open(os.path.join(resume_path, 'features_{}.pickle'.format(split)), 'rb') as f:
        feature = pickle.load(f)
        qFeat = feature['qFeat'].astype('float32')
        dbFeat = feature['dbFeat'].astype('float32')
        gt = feature['gt']
    return gt, qFeat, dbFeat

def load_split(resume_path, split='test'):
    model_parameter_file = os.path.join(resume_path, 'flags.json')
    assert os.path.exists(model_parameter_file), 'model parameter file does not exists, you may have broken your log directory'
    with open(model_parameter_file, 'r') as f:
        opt = json.load(f)
    try:
        db = pd.read_csv(os.path.join(opt['structDir'], 'database.csv'), sep=',')
        val = pd.read_csv(os.path.join(opt['structDir'], '{}.csv'.format(split)), sep=',')
    except (FileNotFoundError):
        db = pd.read_csv(os.path.join('../..', opt['structDir'], 'database.csv'), sep=',')
        val = pd.read_csv(os.path.join('../..', opt['structDir'], '{}.csv'.format(split)), sep=',')
    info_db = db[['index', 'x', 'y']]
    info_db = np.array(info_db)                                                # [index, x, y]
    info_val = val[['index', 'x', 'y']]
    info_val = np.array(info_val)                                              # [index, x, y]

    return info_db, info_val, opt['structDir'], opt['imgDir']


class Measure:
    def __init__(self, gt, qFeat=None, dbFeat=None, preds=None, dists=None) -> None:
        '''
        gt: List[Array]
        qFeat: N1*D
        dbFeat: N2*D
        '''
        self.num_of_cands = 100
        self.gt = gt
        self.qFeat = qFeat
        self.dbFeat = dbFeat

        if preds is None:
            assert qFeat is not None, 'either qFeat or preds should be provided.'
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbFeat)
            self.dists, self.preds = knn.kneighbors(self.qFeat, self.num_of_cands)  # dist: small to large
        else:
            self.preds = preds
            self.dists = dists

        pass

    def get_preds(self):
        # print('preds: ', self.preds.shape)
        # print('dists: ', self.dists.shape)
        return self.preds, self.dists

    def get_recall_at_n(self):
        n_values = [1, 5, 10]
        correct_at_n = np.zeros(len(n_values))
        for qIx, pred in enumerate(self.preds):
            for i, n in enumerate(n_values):
                if np.any(np.in1d(pred[:n], self.gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        recall_at_n = correct_at_n / len(self.preds) * 100.0

        recalls = {}  # make dict for output
        for i, n in enumerate(n_values):
            recalls[n] = recall_at_n[i]

        return recalls

    def get_pr(self, vis=False):
        dists_m = np.around(self.dists[:, 0], 2)                               # (4620,)
        dists_u = np.array(list(set(dists_m)))
        dists_u = np.sort(dists_u)                                             # small > large

        recalls = []
        precisions = []
        for th in tqdm.tqdm(dists_u, ncols=40):
            TPCount = 0
            FPCount = 0
            FNCount = 0
            TNCount = 0
            for index_q in range(self.dists.shape[0]):
                # Positive
                if self.dists[index_q, 0] < th:
                    # True
                    if np.any(np.in1d(self.preds[index_q, 0], self.gt[index_q])):
                        TPCount += 1
                    else:
                        FPCount += 1
                else:
                    if np.any(np.in1d(self.preds[index_q, 0], self.gt[index_q])):
                        FNCount += 1
                    else:
                        TNCount += 1
            assert TPCount + FPCount + FNCount + TNCount == dists_m.shape[0], 'Count Error!'
            if TPCount + FNCount == 0 or TPCount + FPCount == 0:
                # print('zero')
                continue
            recall = TPCount / (TPCount + FNCount)
            precision = TPCount / (TPCount + FPCount)
            recalls.append(recall)
            precisions.append(precision)
        if vis:
            from matplotlib import pyplot as plt
            plt.style.use('ggplot')
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.plot(recalls, precisions)
            ax.set_title('Precision-Recall')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.savefig('pr.png', dpi=200)
        return recalls, precisions


class Measure_Cosine:
    def __init__(self, gt, qFeat=None, dbFeat=None) -> None:
        '''
        gt: List[Array]
        qFeat: N1*D
        dbFeat: N2*D
        '''
        self.num_of_cands = 100
        self.gt = gt
        self.qFeat = qFeat
        self.dbFeat = dbFeat

        cos_mat = cosine_similarity(qFeat, dbFeat)
        preds = []
        dists = []
        for index_r in range(cos_mat.shape[0]):
            ind = np.argsort(cos_mat[index_r, :])
            ind_inv = ind[::-1]
            preds.append(ind_inv[:20])
            dists.append(cos_mat[index_r, ind_inv[:20]])

        self.preds, self.dists = np.array(preds), np.array(dists)

    def get_preds(self):
        print('preds: ', self.preds.shape)
        print('dists: ', self.dists.shape)
        return self.preds, self.dists

    def get_recall_at_n(self):
        n_values = [1, 5, 10, 20]
        correct_at_n = np.zeros(len(n_values))
        for qIx, pred in enumerate(self.preds):
            for i, n in enumerate(n_values):
                if np.any(np.in1d(pred[:n], self.gt[qIx])):
                    correct_at_n[i:] += 1
                    break
        recall_at_n = correct_at_n / len(self.preds) * 100.0

        recalls = {}                                                           # make dict for output
        for i, n in enumerate(n_values):
            recalls[n] = recall_at_n[i]

        return recalls

    def get_pr(self, vis=False):
        dists_u = np.array(list(set(np.around(self.dists[:, 0], 4))))
        dists_u = np.sort(dists_u)                                             # small > large

        recalls = []
        precisions = []
        for th in tqdm.tqdm(dists_u, ncols=40):
            TPCount = 0
            FPCount = 0
            FNCount = 0
            TNCount = 0
            for index_q in range(self.dists.shape[0]):
                # Positive
                if self.dists[index_q, 0] > th:
                    # True
                    if np.any(np.in1d(self.preds[index_q, 0], self.gt[index_q])):
                        TPCount += 1
                    else:
                        FPCount += 1
                else:
                    if np.any(np.in1d(self.preds[index_q, 0], self.gt[index_q])):
                        FNCount += 1
                    else:
                        TNCount += 1
            assert TPCount + FPCount + FNCount + TNCount == self.dists.shape[0], 'Count Error!'
            if TPCount + FNCount == 0 or TPCount + FPCount == 0:
                # print('zero')
                continue
            recall = TPCount / (TPCount + FNCount)
            precision = TPCount / (TPCount + FPCount)
            recalls.append(recall)
            precisions.append(precision)
        if vis:
            from matplotlib import pyplot as plt
            plt.style.use('ggplot')
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.plot(recalls, precisions)
            ax.set_title('Precision-Recall')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            plt.savefig('pr.png', dpi=200)
        return recalls, precisions


def cal_recall_pred(gt, preds):
    n_values = [1, 5, 10, 20]
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(preds):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / len(preds) * 100.0

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]

    return recalls

def cal_recall_feat(gt, qFeat, dbFeat):

    # --------------------------------- use faiss to do NN search -------------------------------- #
    # faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    # faiss_index.add(dbFeat)
    # dists, predictions = faiss_index.search(qFeat, 20)   # the results is sorted

    # -------------------------------- use sklearn to do NN search ------------------------------- #
    knn = NearestNeighbors(n_jobs=1)
    knn.fit(dbFeat)
    dists, predictions = knn.kneighbors(qFeat, 20)

    recalls = cal_recall_pred(gt, predictions)

    return recalls, dists, predictions

def calculate_pr(gt, qFeat=None, dbFeat=None, dists_ext=None, predictions_ext=None):

    # --------------------------------- use faiss to do NN search -------------------------------- #
    # faiss_index = faiss.IndexFlatL2(qFeat_new.shape[1])
    # faiss_index.add(dbFeat_new)
    # dists, predictions = faiss_index.search(qFeat_new, 20)   # the results is sorted

    # -------------------------------- use sklearn to do NN search ------------------------------- #
    if dists_ext is None:
        knn = NearestNeighbors(n_jobs=1)
        knn.fit(dbFeat)
        dists, predictions = knn.kneighbors(qFeat, 1)   # (4620, 1), (4620, 1)
    else:
        dists, predictions = dists_ext, predictions_ext

    dists_min = np.around(dists[:, 0], 2)
    dists_u = np.array(list(set(dists_min)))
    dists_u = np.sort(dists_u)
    # print(len(dists_u))

    recalls = []
    precisions = []
    for th in tqdm.tqdm(dists_u, ncols=40):
        TPCount = 0
        FPCount = 0
        FNCount = 0
        TNCount = 0
        for index_q in range(dists.shape[0]):
            # Positive
            if dists[index_q] < th:
                # True
                if np.any(np.in1d(predictions[index_q], gt[index_q])):
                    TPCount += 1
                else:
                    FPCount += 1
            else:
                if np.any(np.in1d(predictions[index_q], gt[index_q])):
                    FNCount += 1
                else:
                    TNCount += 1
        assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
        if TPCount + FNCount == 0 or TPCount + FPCount == 0:
            # print('zero')
            continue
        recall = TPCount / (TPCount + FNCount)
        precision = TPCount / (TPCount + FPCount)
        recalls.append(recall)
        precisions.append(precision)

    return recalls, precisions


def calculate_pr_utr(gt, dists_ext, predictions_ext):

    dists, predictions = dists_ext, predictions_ext

    dists_min = np.around(dists[:, 0], 4)
    dists_u = np.array(list(set(dists_min)))
    dists_u = np.sort(dists_u)
    # print(len(dists_u))

    recalls = []
    precisions = []
    for th in tqdm.tqdm(dists_u, ncols=40):
        TPCount = 0
        FPCount = 0
        FNCount = 0
        TNCount = 0
        for index_q in range(dists.shape[0]):
            # Positive
            if dists[index_q] > th:
                # True
                if np.any(np.in1d(predictions[index_q], gt[index_q])):
                    TPCount += 1
                else:
                    FPCount += 1
            else:
                if np.any(np.in1d(predictions[index_q], gt[index_q])):
                    FNCount += 1
                else:
                    TNCount += 1
        assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
        if TPCount + FNCount == 0 or TPCount + FPCount == 0:
            # print('zero')
            continue
        recall = TPCount / (TPCount + FNCount)
        precision = TPCount / (TPCount + FPCount)
        recalls.append(recall)
        precisions.append(precision)

    return recalls, precisions
