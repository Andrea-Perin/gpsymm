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
out_dir = Path(cfg['paths']['out_path'])


parser = argparse.ArgumentParser(description='Create comparison plot from various results files')
parser.add_argument('--input_path', type=str, default='results_new',
                    help='Directory to save the plots')
parser.add_argument('--output_path', type=str, default=out_dir / 'comparison',
                    help='Directory to save the plots')
parser.add_argument('--idx', type=int, default=2,
                    help='Index of N to plot (default: 2)')
parser.add_argument('--alpha', type=float, default=.1,
                    help='Transparency of scatter dots (default: .5)')
args = parser.parse_args()

in_dir = Path(args.input_path)
out_dir = Path(args.output_path)
out_dir.mkdir(parents=True, exist_ok=True)


# load files
simple_mlp = np.load(in_dir / 'mlp_1/results.npy')
deep_mlp = np.load(in_dir / 'mlp_5/results.npy')
train_mlp = np.load(in_dir / 'mlp_trained_1/results.npy')
cnn_fc = np.load(in_dir / 'cntk_fc_3/results.npy')
cnn_gap = np.load(in_dir / 'cntk_gap_3/results.npy')

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
data_list = [simple_mlp, train_mlp, deep_mlp, cnn_fc, cnn_gap]
titles = ['1HL MLP', 'MLP (trained)', '5HL MLP', 'CNN-FC', 'CNN-GAP']
xlabs = ['Exact err. (NTK)', 'Emp. err. (trained)', 'Exact err. (NTK)', 'Exact err. (NTK)', 'Exact err. (NTK)']

# Plot in each axis
norm = plt.Normalize(vmin=0, vmax=2)
for idx, (ax, data, title, xlab) in enumerate(zip(grid, data_list, titles, xlabs)):
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
    ax.set_ylabel('Spectral error (NTK)')
    # Set common limits and ticks
    ax.set_xlim([0, 2.5])
    ax.set_xticks([0, 1, 2])
    ax.set_ylim([0, 2.5])
    ax.set_yticks([0, 1, 2])
    ax.plot([0, 2.5], [0, 2.5], color='black', alpha=1, lw=.75, ls='--')

    # add boldface letters
    ax.text(-0.15, 1.185, f'$\\mathbf{{({chr(65+idx)})}}$', transform=ax.transAxes,
            fontsize=10, va='top')

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
