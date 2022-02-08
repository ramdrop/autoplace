from torch.utils.data import DataLoader
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import faiss
import tqdm
import os
import pickle

# efemarai
# import efemarai as ef


def get_recall(opt, model, eval_set, seed_worker, epoch=1, writer=None):
    torch.cuda.set_device(opt.cGPU)
    device = torch.device("cuda")

    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

    model.eval()
    with torch.no_grad():
        dbFeat = np.empty((len(eval_set), opt.output_dim))
        print('get recall..')
        for iteration, (input, indices) in enumerate(tqdm.tqdm(test_data_loader, ncols=40), 1):
            input = input.to(device)
            vlad_encoding = model(input)
            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            del input, vlad_encoding
    del test_data_loader

    n_values = [1, 5, 10, 20]
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')

    # ---------------------------------------------------- sklearn --------------------------------------------------- #
    # knn = NearestNeighbors(n_jobs=-1)
    # knn.fit(dbFeat)
    # dists, predictions = knn.kneighbors(qFeat, len(dbFeat))

    # ----------------------------------------------------- faiss ---------------------------------------------------- #
    faiss_index = faiss.IndexFlatL2(opt.output_dim)
    faiss_index.add(dbFeat)
    # dists, predictions = faiss_index.search(qFeat, max(n_values))   # the results is sorted
    dists, predictions = faiss_index.search(qFeat, len(dbFeat))               # the results is sorted
    # ------------------------------------------------------- - ------------------------------------------------------ #

    # for each query get those within threshold distance
    gt = eval_set.getPositives()
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ * 100.0

    recalls = {}               # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        if writer:
            writer.add_scalar('{}_recall@{}'.format(opt.split, n), recall_at_n[i], epoch)

    if opt.mode == 'evaluate':
        with open(os.path.join(opt.resume, 'features_{}.pickle'.format(eval_set.whichSet)), 'wb') as f:
            feature = {'qFeat': qFeat, 'dbFeat': dbFeat, 'gt': gt}
            pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('saved features_{}.pickle.'.format(eval_set.whichSet))
        print('[{}]\t'.format(opt.split), end='')
        print('recall@1: {:.2f}\t'.format(recalls[1]), end='')
        print('recall@5: {:.2f}\t'.format(recalls[5]), end='')
        print('recall@10: {:.2f}\t'.format(recalls[10]), end='')
        print('recall@20: {:.2f}\t'.format(recalls[20]))

    return recalls
