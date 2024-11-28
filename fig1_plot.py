# %% This does the same thing as the other one, but with easier data loading
from pathlib import Path
from jax import numpy as jnp
import einops as ein

from utils.conf import load_config
from utils.plot_utils import cm, add_inner_title
import matplotlib.pyplot as plt
plt.style.use('myplots.mlpstyle')


# %% Load params and paths
cfg = load_config('config.toml')
N_EPOCHS = cfg['params']['n_epochs']
ANGLES = cfg['params']['rotations']  # [2, 4, 8, 16, 32, 64]
img_path = Path(cfg['paths']['img_path'])
res_path = Path(cfg['paths']['res_path'])
out_dir = img_path / 'fig1'
out_dir.mkdir(parents=True, exist_ok=True)


# %% Load and reshape properly
data = jnp.load(res_path / 'results_all_epochs.npy')
avg_data = ein.reduce(data, 'digit ... -> ...', 'mean')
std_data = ein.reduce(data, 'digit ... -> ...', jnp.std)
print(f'Shape should be: (num_angles, num_models, num_values, num_epochs): {avg_data.shape}')

# %% PANEL B
angle_idx = 2  # corresponding to 8 angles
num_rot = ANGLES[angle_idx]
b_avg = avg_data[angle_idx]
b_std = std_data[angle_idx]
c95 = 1.96*b_std / jnp.sqrt(10)

fig, axs = plt.subplots(
    figsize=(8.5*cm, 10*cm),
    nrows=3, ncols=1,
    sharex=True, sharey=True
)
for ax, avg, std, name in zip(axs.flatten(), b_avg, c95, ('MLP', "CNN", 'ViT')):
    ax.plot(avg[1], color='black')
    ax.plot(avg[2], color='black', ls='--')
    ax.plot(avg[3], ls='--')
    ax.hlines(.1, xmin=0, xmax=N_EPOCHS-1, color='gray', ls='--')
    # add std shades
    # ax.fill_between(jnp.arange(len(avg[1])), avg[1]-std[1], avg[1]+std[1], alpha=.5, color='grey')
    # ax.fill_between(jnp.arange(len(avg[2])), avg[2]-std[2], avg[2]+std[2], alpha=.5, color='grey')
    ax.fill_between(jnp.arange(len(avg[3])), avg[3]-std[3], avg[3]+std[3], alpha=.2)
    ax.set_xlim((0, None))
    ax.set_ylim((0, 1))
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
# plt.savefig(out_dir / 'panelB.pdf', bbox_inches='tight')
plt.show()


# %% PANEL C
data_c = 1-avg_data[:, :, -1, -1]
std_c = std_data[:, :, -1, -1]
c95 = 1.96 * std_c / jnp.sqrt(10)
#std_c = 2*std_c / jnp.sqrt(jnp.array(ANGLES)[:, None])  # TODO: check this one!
print(f'This should be something of shape (n_angles, n_models); {data_c.shape}')

fig, ax = plt.subplots(figsize=(8.5*cm, 5*cm))
ax.set_xscale('log', base=2)
ax.plot(ANGLES, data_c[:, 0], marker='^', label='MLP')
ax.fill_between(ANGLES, data_c[:, 0]-c95[:, 0], data_c[:, 0]+c95[:, 0], alpha=.2)

ax.plot(ANGLES, data_c[:, 1], marker='o', label='CNN')
ax.fill_between(ANGLES, data_c[:, 1]-c95[:, 1], data_c[:, 1]+c95[:, 1], alpha=.2)

ax.plot(ANGLES, data_c[:, 2], marker='s', label='ViT')
ax.fill_between(ANGLES, data_c[:, 2]-c95[:, 2], data_c[:, 2]+c95[:, 2], alpha=.2)
ax.set_xticks(ANGLES)
ax.set_xticklabels(ANGLES)
ax.set_xlabel(r'$N$')
ax.set_ylabel('Test (out) error')
ax.set_ylim((0, 1))
ax.legend()
plt.tight_layout()
# plt.savefig(out_dir / 'panelC.pdf')
plt.show()
# %%
mlp_errs = data[:, -1, 0, -1]
for n, digit in enumerate(mlp_errs):
    plt.plot(digit, label=n)
plt.title('test out accuracy')
plt.legend()
plt.show()
# %%
print(data.shape)
# %%
mlp_train_acc = data[:, -1, 0, 1]
for n, digit in enumerate(mlp_train_acc):
    plt.plot(digit, label=n)
plt.legend()
plt.show()
