# %%
import jax
from jax import numpy as jnp
import numpy as np
import einops as ein
from jaxlib.xla_client import Layout
from jaxtyping import Float, Array
import functools as ft
from pathlib import Path

from utils.gp_utils import kreg, extract_components, circulant_error, make_circulant
from utils.plot_utils import cm, get_size

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
plt.style.use('./myplots.mlpstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bcmap = matplotlib.colors.ListedColormap([colors[0], colors[1]])

out_path = Path('images/toy_model')
out_path.mkdir(parents=True, exist_ok=True)

# we use the orthogonal normalization in our computations.
# This means that sometimes we will introduce sqrt(N) factors around.
# See https://numpy.org/devdocs/reference/routines.fft.html#normalization
orthofft = ft.partial(jnp.fft.fft, norm='ortho')


def get_data(N: int, d: float, r: float=1.,) -> Float[Array, '2N 3']:
    angles_p = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
    angles_m = angles_p + jnp.pi/N
    xs_p, _ = ein.pack( (r*jnp.cos(angles_p), r*jnp.sin(angles_p), jnp.ones(N)*d/2), 'd *' )
    xs_m, _ = ein.pack( (r*jnp.cos(angles_m), r*jnp.sin(angles_m), -d/2*jnp.ones(N)), 'd *' )
    data, ps = ein.pack((xs_p, xs_m), '* d')
    return ein.rearrange(data, '(s n) d -> (n s) d', n=N)


# %% PANELS A, B, C: data
Ns = range(4, 50, 4)
ratio = .2/3
d = 1
L = 1.4
r = 1.

get_k = lambda data: jnp.exp(-L**2 * jnp.sum((data-data[:, None])**2, axis=-1))
datasets = [get_data(n, d, r) for n in Ns]
ys = [jnp.array([-1., 1.]*n)[:, None] for n in Ns]
kernels = [get_k(data) for data in datasets]
reg = 1e-5


errors, our_errors = [], []
for k, y, n in zip(kernels, ys, Ns):
    # empirical errors
    k_train_train, k_train_test, k_test_test = extract_components(k, n//2)
    y_train = jnp.delete(y, n//2, axis=0)
    ymean, yvar = kreg(k_train_train, k_train_test, k_test_test, y_train, reg=reg)
    errors.append(jnp.abs(y[n//2] - ymean))
    # spectral errors
    our_errors.append(circulant_error(k, reg=reg))
errors = jnp.array(errors).flatten()
our_errors = jnp.array(our_errors).flatten()


# %% PANEL B
fig, ax = plt.subplots(figsize=(5*cm, 8*cm))
ax.plot(Ns, jnp.log(errors), 'o-', color='black', label=r'Emp.')
ax.plot(Ns, jnp.log(our_errors), '.-', color=colors[0], label=r'Ours')
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$\log(\varepsilon)$', ha='left', rotation=0, y=1, labelpad=0)
ax.set_xticks([Ns[0], 25, 50])
ax.set_title(', '.join((f'$L={L}$', f'$\Delta={d}$')))
ax.legend()
plt.tight_layout()
plt.savefig(out_path / 'case2_panelB.pdf')
plt.show()


# %% AC PANELS TOGETHER
idxs_to_plot = [0, 1, 4]
figsize=(12*cm, 8*cm)
fig = plt.figure(figsize=figsize)
hmult = 5

gs0 = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], wspace=0, hspace=0)
gs_bot = GridSpecFromSubplotSpec(1, 3*hmult, subplot_spec=gs0[1], wspace=0, hspace=0)


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


# Top row: 3D projections with no ticks but axis labels
for axidx, idx in enumerate(idxs_to_plot):
    xdata = datasets[idx]
    ax = fig.add_subplot(gs_top[axidx], projection='3d', computed_zorder=False)
    ax = clean_3d_ax(ax)
    ax.set_title(f"$N = {len(xdata)//2}$")
    # data
    ax.scatter(xdata[1::2, 2], xdata[1::2, 1], xdata[1::2, 0], c=colors[1], marker='o', depthshade=False)
    ax.scatter(xdata[2::2, 2], xdata[2::2, 1], xdata[2::2, 0], c=colors[0], marker='o', depthshade=False)
    ax.scatter(xdata[0, 2], xdata[0, 1], xdata[0, 0], c='black', depthshade=False, zorder=10)
    # add fake circle
    theta = jnp.linspace(0, 2 * jnp.pi, 201)
    d = jnp.abs(xdata[0, -1])*2
    ax.plot(jnp.ones_like(theta)*d/2, jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[0], lw=.5)
    ax.plot(-d/2*jnp.ones_like(theta), jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[1], lw=.5)
    # labels
    ax.text2D(0.25, 0.05, "x", transform=ax.transAxes)
    ax.text2D(0.85, 0.15, "y", transform=ax.transAxes)
    ax.text2D(0.95, 0.85, "z", transform=ax.transAxes)

# Bottom row: 2D plots with shared y-axis and short ytick labels
for axidx, idx in enumerate(idxs_to_plot):
    ax = fig.add_subplot(gs_bot[axidx*5+1:(axidx+1)*5-1])
    k = kernels[idx]
    n, last = len(k), len(k)//2
    isp = 1/jnp.abs(orthofft(k[0]) + reg)
    #
    isp = (isp/isp.max())[:last+1]
    ws = jnp.arange(last+1)
    ax.plot(ws, isp, color=colors[0])
    ax.fill_between(ws, 0, isp, color=colors[0], alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='grey', linestyle='--', alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='None', edgecolor='grey', linestyle='--', alpha=.5, hatch='/////')
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, last], ['0', 'N'])
    # add text indicating last eigenvalue
    if axidx==2:
        axins = ax.inset_axes(
            ( 0.6, 0.6, 0.6, 0.6 ),
            xlim=(last-3, last+.2), ylim=(0, isp[-4]),
            xticklabels=[], yticklabels=[])
        axins.plot(ws[-4:], isp[-4:], color=colors[0])
        axins.fill_between(ws[-4:], 0, isp[-4:], color=colors[0], alpha=.5)
        axins.fill_between(ws[-4:], 0, isp[-1], color='grey', linestyle='--', alpha=.5)
        axins.fill_between(ws[-4:], 0, isp[-1], color='None', edgecolor='grey', linestyle='--', alpha=.5, hatch='/////')
        axins.text(float(last*1.01), float(isp[-1]), r'$\lambda^{-1}_N$')
        axins.scatter(last, isp[-1], c='grey', s=3, zorder=10)
        ax.indicate_inset_zoom(axins, edgecolor="black")
    else:
        ax.text(float(last*1.1), float(isp[-1]), r'$\lambda^{-1}_N$')
        ax.scatter(last, isp[-1], c='grey', s=3, zorder=10)

fig.supylabel(r'$\lambda^{-1}$', x=0.1, y=.3, ha='center')
fig.supxlabel('Frequency', ha='center', y=0.09)
plt.tight_layout()
plt.savefig(out_path / 'case2_panelAC.pdf')
plt.show()


# %% PANELS D, E, F: data
N = 6
L = .85
r = 1.
ds = jnp.array([0.] + list(jnp.logspace(-2, 1, 20)))
reg = 1e-5

get_k = lambda data: jnp.exp(-L**2 * jnp.sum((data-data[:, None])**2, axis=-1))
datasets = [get_data(N, d, r) for d in ds]
ys = jnp.array([-1., 1.]*N)[:, None]
kernels = [get_k(data) for data in datasets]

errors, our_errors = [], []
for k in kernels:
    # empirical errors
    k_train_train, k_train_test, k_test_test = extract_components(k, N//2)
    y_train = jnp.delete(ys, N//2, axis=0)
    ymean, yvar = kreg(k_train_train, k_train_test, k_test_test, y_train, reg=reg)
    errors.append(jnp.abs(ys[N//2] - ymean))
    # spectral errors
    our_errors.append(circulant_error(k, reg=reg))
errors = jnp.array(errors).flatten()
our_errors = jnp.array(our_errors).flatten()


# %% DF PANELS TOGETHER, BUT NICE
idxs_to_plot = [0, 12, 15]
figsize=(12*cm, 8*cm)
fig = plt.figure(figsize=figsize)
hmult = 5

gs0 = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], wspace=0, hspace=0)
gs_bot = GridSpecFromSubplotSpec(1, 3*hmult, subplot_spec=gs0[1], wspace=0, hspace=0)

# Top row: 3D projections with no ticks but axis labels
for axidx, idx in enumerate(idxs_to_plot):
    xdata = datasets[idx]
    ax = fig.add_subplot(gs_top[axidx], projection='3d', computed_zorder=False)
    ax = clean_3d_ax(ax)
    ax.set_title(f"$\Delta = {ds[idx]:.2f}$")
    # data
    ax.scatter(xdata[1::2, 2], xdata[1::2, 1], xdata[1::2, 0], c=colors[1], marker='o', depthshade=False)
    ax.scatter(xdata[2::2, 2], xdata[2::2, 1], xdata[2::2, 0], c=colors[0], marker='o', depthshade=False)
    ax.scatter(xdata[0, 2], xdata[0, 1], xdata[0, 0], c='black', depthshade=False, zorder=10)
    # add fake circle
    theta = jnp.linspace(0, 2 * jnp.pi, 201)
    d = jnp.abs(xdata[0, -1])*2
    ax.plot(jnp.ones_like(theta)*d/2, jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[0], lw=.5)
    ax.plot(-d/2*jnp.ones_like(theta), jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[1], lw=.5)
    # labels
    ax.text2D(0.25, 0.05, "x", transform=ax.transAxes)
    ax.text2D(0.85, 0.15, "y", transform=ax.transAxes)
    ax.text2D(0.95, 0.85, "z", transform=ax.transAxes)
    # lims
    ax.set_xlim((-1, 1))

# Bottom row: 2D plots with shared y-axis and short ytick labels
for axidx, idx in enumerate(idxs_to_plot):
    ax = fig.add_subplot(gs_bot[axidx*5+1:(axidx+1)*5-1])
    k = kernels[idx]
    n, last = len(k), len(k)//2
    isp = 1/jnp.abs(orthofft(k[0]) + 1e-5)
    isp = (isp/jnp.max(isp))[:last+1]
    ws = jnp.arange(last+1)
    ax.plot(ws, isp, color=colors[0])
    ax.fill_between(ws, 0, isp, color=colors[0], alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='grey', linestyle='--', alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='None', edgecolor='grey', linestyle='--', alpha=.5, hatch='/////')
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, last], ['0', 'N'])
    # add text indicating last eigenvalue
    ax.text(float(last*(1.1)), float(isp[-1]), r'$\lambda^{-1}_N$')
    ax.scatter(last, isp[-1], c='grey', s=3, zorder=10)

fig.supylabel(r'$\lambda^{-1}$', x=0.1, y=.3, ha='center')
fig.supxlabel('Frequency', ha='center', y=0.09)
plt.tight_layout()
plt.savefig(out_path / 'case1_panelAC.pdf')
plt.show()


# %% PANEL E
figsize = (5*cm, 8*cm)
fig, ax = plt.subplots(figsize=figsize)
ax.plot(jnp.log(ds), jnp.log(errors), 'o-', color='black', label=r'Emp.')
ax.plot(jnp.log(ds), jnp.log(our_errors), '.-', color=colors[0], label=r'Ours')
ax.set_xlabel(r'$\log\Delta$')
ax.set_ylabel(r'$\log(\varepsilon)$', ha='left', rotation=0, y=1, labelpad=0)
ax.set_title(', '.join((f'$L={L}$', f'$N={N}$')))
ax.legend()
plt.tight_layout()
plt.savefig(out_path / 'case1_panelB.pdf')
plt.show()
