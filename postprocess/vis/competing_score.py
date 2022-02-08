# %%
import numpy as np
import pickle
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


groups = ['se_te_dpr_rcshr', 'netvlad', 'netvlad_lstm', 'seqnet', 'minkloc3d', 'm2dp', 'scancontext', 'kidnapped', 'utr']
labels = ['AutoPlace', 'NetVLAD', 'NetVLAD+LSTM', 'SeqNet', 'MinkLoc3D', 'M2DP', 'ScanContext', 'KidnappedRadar', 'UnderTheRadar']

# mask = [0, 1, 2, 3, 4, 5, 6]
# groups = [groups[m] for m in mask]
# labels = [labels[m] for m in mask]

recalls_list = []
precisions_list = []
recalls_at_n_list = []
for index, g in enumerate(groups):
    print(g)
    with open('./../parse/results/{}_results.pickle'.format(g), 'rb') as f:
        feature = pickle.load(f)
        precisions_list.append(feature['precisions'])
        recalls_list.append(feature['recalls'])
        recalls_at_n_list.append(feature['recalls_at_n'])

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

with open('figures/competing/scores-competing.txt', 'w') as f:
    f.write('{:>10s}\t{:>4s}\t{:>4s}\n'.format('Recall@1/5/10', 'F1', 'AP'))
    for index, recall in enumerate(recalls_at_n_list):
        f.write('{:.1f}/{:.1f}/{:.1f} & {:.2f} & {:.2f}\t {:<10s}\n'.format(recall[1], recall[5], recall[10], f1s_list[index], ap_list[index], labels[index]))
