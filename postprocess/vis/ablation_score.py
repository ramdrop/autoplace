# %%
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from os.path import join, exists
from os import mkdir
import numpy as np
import pickle

# ------------------------------------------- Recall@N ------------------------------------------- #
duts = ['se', 'se_te', 'se_dpr', 'se_te_dpr', 'se_te_dpr_rcshr', 'se_rcshr', 'se_dpr_rcshr', 'se_te_rcshr']
labels = ['SE', 'SE+TE', 'SE+DPR', 'SE+TE+DPR', 'SE+TE+DPR+RCSHR', 'SE+RCSHR', 'SE+DPR+RCSHR', 'SE+TE+RCSHR']

# mask = [0, 1, 2, 3, 4]
# mask = [0, 1, 2, 3, 4]
# duts = [duts[m] for m in mask]
# labels = [labels[m] for m in mask]

if not exists('figures'):
    mkdir('figures')
    mkdir('figures/ablation')

recalls_at_n_list = []
recalls_list = []
precisions_list = []

for dut in duts:
    with open('./../parse/results/{}_results.pickle'.format(dut), 'rb') as f:
        feature = pickle.load(f)
        recalls_at_n = feature['recalls_at_n']
        recalls = feature['recalls']
        precisions = feature['precisions']
    recalls_at_n_list.append([recalls_at_n[1], recalls_at_n[5], recalls_at_n[10]])
    recalls_list.append(recalls)
    precisions_list.append(precisions)

# ------------------------------------------- F1 Score ------------------------------------------- #
f1s_list = []
for index in range(len(recalls_list)):
    recalls = np.array(recalls_list[index])
    precisions = np.array(precisions_list[index])
    ind = np.argsort(recalls)
    recalls = recalls[ind]
    precisions = precisions[ind]
    f1s = []
    for index_j in range(len(recalls)):
        f1 = 2 * precisions[index_j] * recalls[index_j] / (precisions[index_j] + recalls[index_j])
        f1s.append(f1)
    f1s_list.append(max(f1s))
print('F1 Score:', f1s_list)


# --------------------------------------- Average Precision -------------------------------------- #
# refer to -> Evaluation of Object Proposals and ConvNet Features for Landmark-based Visual Place Recognition
ap_list = []
for index in range(len(recalls_list)):
    recalls = np.array(recalls_list[index])
    precisions = np.array(precisions_list[index])
    ind = np.argsort(recalls)
    recalls = recalls[ind]
    precisions = precisions[ind]
    ap = 0
    for index_j in range(len(recalls) - 1):
        ap += precisions[index_j] * (recalls[index_j + 1] - recalls[index_j])
    ap_list.append(ap)
print('Average Precision:', ap_list)

with open('figures/ablation/scores-ablation.txt', 'w') as f:
    f.write('{:>10s}\t{:>4s}\t{:>4s}\n'.format('Recall@1/5/10', 'F1', 'AP'))
    for index, recall in enumerate(recalls_at_n_list):
        # f.write('{:.1f},{:.1f}, {:.1f}\t {:.4f}\t{:.4f}\t {:<10s}\n'.format(recall[0], recall[1], recall[2], f1s_list[index], ap_list[index],
        # labels[index]))
        f.write('{:.1f}/{:.1f}/{:.1f} & {:.2f} & {:.2f}\t {:<10s}\n'.format(recall[0], recall[1], recall[2], f1s_list[index], ap_list[index], labels[index]))
