# %% Figure 1: the plots without the data generating code
import re
import csv
from pathlib import Path
import numpy as np
import einops as ein

from plot_utils import cm, add_inner_title

import matplotlib.pyplot as plt
plt.style.use('myplots.mlpstyle')

model_name_to_int = {'mlp': '0', 'conv': '1', 'vit': '2'}
int_to_model_name = {v:k for k, v in model_name_to_int.items()}


def read_data(results_path: Path | str) -> list:
    contents = []
    with open(results_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for line in reader:
            contents.append(line[2:])
    return contents


def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


plot_dir = Path('./images/fig1')
plot_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path('./results')
results = map(str, results_dir.iterdir())
sorted_results = sorted(results, key=natural_key)
all_data = sum(map(read_data, sorted_results), start=[])
all_data = np.array(all_data, dtype=float)
# insert angle information by hand; assumes a single number in each filename
angles = np.array([ int(re.split(r'(\d+)', res_name)[1]) for res_name in sorted_results ])
all_data = ein.rearrange(
    all_data,
    '(nrot epoch model) values -> nrot epoch model values',
    nrot=len(angles), model=3
)

# %% PANEL B
fig, axs = plt.subplots(
    figsize=(8.5*cm, 10*cm),
    nrows=3, ncols=1,
    sharex=True, sharey=True
)
data_b = ein.rearrange(all_data[2], 'epoch model values -> model epoch values')
for ax, vals, name in zip(axs.flatten(), data_b, ('MLP', "CNN", 'ViT')):
    ax.plot(vals[:, 1], color='black')
    ax.plot(vals[:, 2], color='black', ls='--')
    ax.plot(vals[:, 3], ls='--')
    ax.hlines(.1, xmin=0, xmax=len(vals)-1, color='gray', ls='--')
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))
    at = add_inner_title(ax, name, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.25,rounding_size=0.2")
    at.patch.set_edgecolor('grey')
    # ax.text(.75, .5, name, transform=ax.transAxes)
    ax.set_ylabel('Accuracy')
    if name == 'ViT':
        ax.set_xlabel('Epoch')
labels = ['Train', 'Test (in)', 'Test (out)', 'Chance']
fig.legend(labels=labels,
    loc='upper center',
    bbox_to_anchor=(0.45, 0.98),
    ncol=4,
    bbox_transform=fig.transFigure,
    handlelength=1,        # Reduced from 1
    columnspacing=0.4,       # Reduced from 0.5
    handletextpad=0.2,       # Added to reduce space between handle and text
    borderaxespad=0.1)       # Optional: reduces padding around the legend
plt.subplots_adjust(top=0.85, bottom=0.1, wspace=0, hspace=0.2)
plt.savefig(plot_dir / 'panelB.pdf', bbox_inches='tight')
plt.show()
# %% PANEL C
data_c = 1-all_data[:, -1, :, -1]
print(f'This should be something of shape (n_angles, n_models); {data_c.shape}')

fig, ax = plt.subplots(figsize=(8.5*cm, 5*cm))
ax.set_xscale('log', base=2)
ax.plot(angles, data_c[:, 0], marker='^', label='MLP')
ax.plot(angles, data_c[:, 1], marker='o', label='CNN')
ax.plot(angles, data_c[:, 2], marker='s', label='ViT')
ax.set_xticks(angles)
ax.set_xticklabels(angles)
ax.set_xlabel(r'$N_{{rot}}$')
ax.set_ylabel('Test (out) error')
ax.set_ylim((0, 1))
ax.legend()
plt.tight_layout(pad=0)
plt.savefig(plot_dir / 'panelC.pdf')
plt.show()
