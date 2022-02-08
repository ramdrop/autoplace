# %%
import matplotlib.pyplot as plt
import pickle
import matplotlib.font_manager as fm
from os.path import join, exists
from os import mkdir

font_path = 'arial.ttf'
my_font = fm.FontProperties(fname=font_path, size=12)

if not exists('figures'):
    mkdir('figures')
    mkdir('figures/ablation')


# duts = ['encoder', 'encoder_lstm', 'encoder_dtr', 'encoder_dtr_lstm', 'encoder_dtr_lstm_rcshr', 'encoder_rcshr', 'encoder_dtr_rcshr', 'encoder_lstm_rcshr']
# labels = [
#     'SpatialEncoder', 'SpatialEncoder\n+TemporalEncoder', 'Encoder+DTR', 'SpatialEncoder\n+TemporalEncoder\n+DPR', 'SpatialEncoder\n+TemporalEncoder\n+DPR\n+RCSHR',
#     'Encoder+RCSHR', 'Encoder+DTR+RCSHR', 'Encoder+LSTM+RCSHR'
# ]

duts = ['se', 'se_te', 'se_dpr', 'se_te_dpr', 'se_te_dpr_rcshr', 'se_rcshr', 'se_dpr_rcshr', 'se_te_rcshr']
labels = ['SE', 'SE+TE', 'SE+DPR', 'SE+TE+DPR', 'SE+TE+DPR+RCSHR', 'SE+RCSHR', 'SE+DPR+RCSHR', 'SE+TE+RCSHR']

# mask = [0, 1, 3, 4]
# mask = [0, 1, 3, 4]
group_num = 4
if group_num == 1:
    mask = [0]
elif group_num == 2:
    mask = [0, 1]
elif group_num == 3:
    mask = [0, 1, 3]
elif group_num == 4:
    mask = [0, 1, 3, 4]

duts = [duts[m] for m in mask]
labels = [labels[m] for m in mask]
LW = 2

recalls_at_n_list = []
recalls_list = []
precisions_list = []
for dut in duts:
    with open('../parse/results/{}_results.pickle'.format(dut), 'rb') as f:
        feature = pickle.load(f)
        recalls_at_n = feature['recalls_at_n']
        recalls = feature['recalls']
        precisions = feature['precisions']
    recalls_at_n_list.append([recalls_at_n[1], recalls_at_n[5], recalls_at_n[10]])
    recalls_list.append(recalls)
    precisions_list.append(precisions)

markers = [
    'o',
    'p',
    's',
    '^',
    '*',
    'o',
    'p',
    's']
linestyles = [
    '-',
    '--',
    ':',
    '-',
    '-.',
    '-',
    '--',
    ':']

# ------------------------------------------- Recall@N ------------------------------------------- #
plt.style.use('ggplot')
fig = plt.figure(figsize=(5.5, 5))
ax = fig.add_subplot(111)
for index, recall in enumerate(recalls_at_n_list):
    ax.plot([1, 5, 10], [recall[0], recall[1], recall[2]],
    lw=LW,
    # linestyle=linestyles[index],
    label=labels[index])
ax.set_xticks([1, 5, 10])
ax.set_xlabel('N', fontproperties=my_font)
ax.set_ylabel('Recall@N(%)', fontproperties=my_font)
ax.set_ylim([72, 86])
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), prop=my_font)
legend = ax.legend(loc='lower right', prop=my_font)
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')
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
plt.tight_layout()
ax.grid('on', color='#e6e6e6')
for label in ax.get_xticklabels():
    label.set_fontproperties(my_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(my_font)
plt.savefig('figures/ablation/recall_at_n-ablation_{}.png'.format(group_num), dpi=200)
plt.savefig("figures/ablation/recall_at_n-ablation.svg", format="svg")

# ---------------------------------------------- PR ---------------------------------------------- #
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
for index in range(len(recalls_list)):
    ax.plot(
        recalls_list[index],
        precisions_list[index],
               # linestyle=linestyles[index],
        lw=LW,
        label=labels[index])
ax.set_xlabel('Recall', fontproperties=my_font)
ax.set_ylabel('Precision', fontproperties=my_font)
ax.set_ylim([0.70, 1.01])

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), prop=my_font)
legend = ax.legend(loc='lower left', prop=my_font)
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')
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
plt.tight_layout()
ax.grid('on', color='#e6e6e6')
for label in ax.get_xticklabels():
    label.set_fontproperties(my_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(my_font)

plt.savefig('figures/ablation/PR-ablation_{}.png'.format(group_num), dpi=200)
plt.savefig('figures/ablation/PR-ablation.svg', format="svg")
