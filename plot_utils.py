import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke

# Define some constants to be shared by the plots
cm = 1/2.54


def get_size(margin: float = 2*cm, aspect: float = 1.618):
    width = 21*cm - 2*margin
    return (width, width/1.618)


def cloudplot(xvals, yvals, cvals, xlabel, ylabel, clabel, titles, figsize):
    assert len(xvals) == len(yvals) == len(cvals) == len(titles)
    n_rots = len(xvals)
    fig = plt.figure(figsize=figsize)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, n_rots), axes_pad=0.1, label_mode="L", share_all=True,
        cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.1, aspect=False)
    for ax, xs, ys, cs, title in zip(grid, xvals, yvals, cvals, titles):
        ax.set_title(title)
        im = ax.scatter(xs, ys, marker='.', alpha=.1, s=2, c=cs)
    # colorbar
    cbar = ax.cax.colorbar(plt.cm.ScalarMappable(cmap=im.get_cmap(), norm=im.norm))
    cbar.set_label(clabel)
    return fig


def add_inner_title(ax, title, loc, **kwargs):
    prop = dict(path_effects=[withStroke(foreground='w', linewidth=3)],
                size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=True, **kwargs)
    ax.add_artist(at)
    return at


def add_spines(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])


def clean_3d_ax(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Remove ticks from each axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    return ax
