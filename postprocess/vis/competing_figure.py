# %%
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from os.path import join, exists
from os import mkdir
import pathlib


os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.font_manager as fm
font_path = 'arial.ttf'
my_font = fm.FontProperties(fname=font_path, size=12)

pathlib.Path('figures/competing').mkdir(parents=True, exist_ok=True)

groups = ['se_te_dpr_rcshr', 'netvlad', 'netvlad_lstm', 'seqnet',  'minkloc3d', 'm2dp', 'scancontext', 'kidnapped', 'utr']
labels = ['Ours:AutoPlace', 'NetVLAD', 'NetVLAD+LSTM', 'SeqNet',  'MinkLoc3D', 'M2DP', 'ScanContext', 'KidnappedRadar', 'UnderTheRadar']
mask = [0, 1, 3, 4, 5, 6, 7, 8]
groups = [groups[m] for m in mask]
labels = [labels[m] for m in mask]
markers = ['o', 'p', 's', '^', 'o', 'p', 's', '^', 'o', 'p', 's', 'o']
dot_p = 0.3
dot_n = 1.5
line_p = 2
line_n = 1.5
linestyles = [
    'solid', (0, (1, 1.5, 1, 1.5)), (0, (1, 3, 1, 3)), (0, (3, 3)), (0, (line_p, line_n, dot_p, dot_n)), (0, (line_p, line_n, line_p, line_n, dot_p, dot_n)),
    (0, (dot_p, dot_n, dot_p, dot_n, line_p, line_n)), (0, (dot_p, dot_n)), (0, (dot_p, dot_n, dot_p, dot_n, dot_p, dot_n, line_p, line_n))
]

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


# ---------------------------------------------- PR ---------------------------------------------- #
plt.style.use('ggplot')
fig = plt.figure(figsize=(4.8, 4.8))
ax = fig.add_subplot(111)
for index in range(len(recalls_list)):
    # ax.plot(recalls_list[index], precisions_list[index], linestyle=linestyles[index], label=labels[index], dash_capstyle='round', solid_capstyle="round")
    ax.plot(recalls_list[index], precisions_list[index], linestyle=linestyles[index], label=labels[index], dash_capstyle='round', solid_capstyle="round", lw=2)
ax.set_xlabel('Recall', fontproperties=my_font)
ax.set_ylabel('Precision', fontproperties=my_font)
# https://www.coder.work/article/93874
# legend = ax.legend(loc='lower left', handlelength=2, prop=my_font)
# legend = ax.legend(loc='upper center', ncol=3, handlelength=1.5, bbox_to_anchor=(0.5, -0.15), borderpad=0.5, labelspacing=0.3, columnspacing=0.3, prop=my_font)
legend = ax.legend(loc='lower left', ncol=2, handlelength=1.8, borderpad=0.5, labelspacing=0.5, columnspacing=1, prop=my_font)
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')

ax.grid('on', color='#e6e6e6')
for label in ax.get_xticklabels():
    label.set_fontproperties(my_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(my_font)
ax.set_facecolor('white')
bwith = 1
ax.spines['top'].set_color('grey')
ax.spines['right'].set_color('grey')
ax.spines['bottom'].set_color('grey')
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.xaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='y', colors='black')
# ax.set_ylim([0.7, 1])
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
# ax.set_aspect(0.7 / ax.get_data_ratio(), adjustable='box')
# ax.set_aspect('equal', 'box')

plt.tight_layout()
plt.savefig("figures/competing/PR-competing.svg", format="svg")
plt.savefig('figures/competing/PR-competing.png', dpi=200)
# plt.savefig("competing/PR-competing.eps", format="eps")
# pp = PdfPages('competing/PR-competing.pdf')
# pp.savefig()
# pp.close()


# ------------------------------------------- Recall@N ------------------------------------------- #
# cmap = plt.get_cmap('cubehelix')
# colors = [cmap(i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
plt.style.use('ggplot')
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
for index, recall in enumerate(recalls_at_n_list):
    ax.plot([1, 5, 10], [recall[1], recall[5], recall[10]], markersize=5, marker=markers[index], linestyle=linestyles[index], label=labels[index])
ax.set_xticks([1, 5, 10])
ax.set_xlabel('N', fontproperties=my_font)
ax.set_ylabel('Recall@N', fontproperties=my_font)
# ax.set_ylim([0, 90])
ax.grid('on')
for label in ax.get_xticklabels():
    label.set_fontproperties(my_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(my_font)
ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.15), prop=my_font)
plt.tight_layout()
plt.savefig('figures/competing/recall_at_n-competing.svg', format='svg')
plt.savefig('figures/competing/recall_at_n-competing.png', dpi=200)
# pp = PdfPages('competing/recall_at_n-competing.pdf')
# pp.savefig()
# pp.close()
