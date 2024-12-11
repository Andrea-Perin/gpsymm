# %% Plot comparison between methods
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


from utils.conf import load_config
from utils.plot_utils import cm, add_inner_title, semaphore, format_axis_scientific
plt.style.use('myplots.mlpstyle')


cfg = load_config()
res_dir = Path(cfg['paths']['res_path'])
out_dir = Path(cfg['paths']['out_path'])


parser = argparse.ArgumentParser(description='Create comparison plot from various results files')
parser.add_argument('--output_path', type=str, default=out_dir / 'comparison',
                    help='Directory to save the plots')
parser.add_argument('--idx', type=int, default=2,
                    help='Index of N to plot (default: 2)')
parser.add_argument('--alpha', type=float, default=.1,
                    help='Transparency of scatter dots (default: .5)')
parser.add_argument('--shift', action='store_true',
                    help='whether to use shifts (default: False)')
args = parser.parse_args()


out_dir = Path(args.output_path)
out_dir.mkdir(parents=True, exist_ok=True)


# load files
simple_mlp = np.load(res_dir / 'mlp_1/results.npy')
deep_mlp = np.load(res_dir / 'mlp_5/results.npy')
train_mlp = np.load(res_dir / 'mlp_trained_1/results.npy')
cnn_fc = np.load(res_dir / 'cntk_fc/results.npy')
cnn_gap = np.load(res_dir / 'cntk_gap/results.npy')


# Set up figure
figsize = (17*cm, 5*cm)  # Adjust size as needed
fig = plt.figure(figsize=figsize)
grid = ImageGrid(fig, 111,
                nrows_ncols=(1, 5),
                axes_pad=0.15,
                share_all=True,
                aspect=True,  # Forces square aspect ratio
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.15,
                )

# List of data and titles
data_list = [simple_mlp, deep_mlp, train_mlp, cnn_fc, cnn_gap]
titles = ['1HL MLP', '5HL MLP', 'Trained 1HL MLP', 'CNN-FC', 'CNN-GAP']
xlabs = ['Emp. err. (NTK)', 'Emp err. (NTK)', 'Emp err. (train)', 'Emp err. (NTK)', 'Emp err. (NTK)']

# Plot in each axis
norm = plt.Normalize(vmin=0, vmax=2)
for ax, data, title, xlab in zip(grid, data_list, titles, xlabs):
    sc = ax.scatter(
        data[2, args.idx],
        data[1, args.idx],
        c=data[-1, args.idx],
        marker='.',
        alpha=args.alpha,
        s=2,
        cmap=semaphore,
        norm=norm
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab)
    ax.set_ylabel('Spectral error')
    # Set common limits and ticks
    ax.set_xlim([0, 2])
    ax.set_xticks([0, 1, 2])
    ax.set_ylim([0, 2])
    ax.set_yticks([0, 1, 2])
    ax.plot([0, 2], [0, 2], color='black', alpha=1, lw=.75, ls='--')
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))


cbar = grid.cbar_axes[0].colorbar(
    plt.cm.ScalarMappable(norm, cmap=semaphore),
    label='Num. correct',
    ticks=[1/3, 1, 1+2/3],
    alpha=1.0
)
cbar.ax.tick_params(size=0)
cbar.ax.set_yticklabels((0, 1, 2), va='center')
plt.tight_layout(pad=0.4)
plt.savefig(out_dir / f'comparison_{args.idx}.pdf')
plt.close()
