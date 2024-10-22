# %%
import jax
from jax import numpy as jnp
import einops as ein
from jaxlib.xla_client import Layout
from jaxtyping import Float, Array
from numpy._typing import _16Bit
from gp_utils import kreg
import functools as ft

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
plt.style.use('./myplots.mlpstyle')
from plot_utils import cm, get_size
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bcmap = matplotlib.colors.ListedColormap([colors[0], colors[1]])


orthorfft = ft.partial(jnp.fft.rfft, norm='ortho')

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

errors = []
for k, y, n in zip(kernels, ys, Ns):
    k_train_train = jnp.delete(jnp.delete(k, n//2, 1), n//2, 0)
    k_train_test = jnp.delete(k[:, n//2:n//2+1], n//2, 0)
    k_test_test = k[n//2:n//2+1, n//2:n//2+1]
    y_train = jnp.delete(y, n//2, axis=0)
    ymean, yvar = kreg(k_train_train, k_train_test, k_test_test, y_train)
    errors.append(jnp.abs(y[n//2] - ymean))
# %% PANEL A
def add_3d_scatter_with_custom_axes(fig, xdata, ydata, magicnum, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    """
    Adds a 3D scatter plot to a given figure with custom axes going through the origin (0, 0, 0).

    Parameters:
    - fig : matplotlib figure object where the scatter plot will be added.
    - data : numpy array of shape (npts, 3), containing the (x, y, z) points.
    """
    # Check if data has the correct shape
    assert data.shape[1] == 3, "Data must have shape (npts, 3)"
    n = len(data) // 2
    # Create a 3D axis
    ax = fig.add_subplot(magicnum, projection='3d', computed_zorder=False)
    # Scatter plot the data: two classes separately
    ax.scatter(xdata[1::2, 2], xdata[1::2, 1], xdata[1::2, 0], c=colors[1], marker='x', depthshade=False)
    ax.scatter(xdata[2::2, 2], xdata[2::2, 1], xdata[2::2, 0], c=colors[0], marker='o', depthshade=False)
    ax.scatter(xdata[0, 2], xdata[0, 1], xdata[0, 0], c='black', depthshade=False, zorder=10000)
    # add fake circle
    theta = jnp.linspace(0, 2 * jnp.pi, 201)
    d = jnp.abs(xdata[0, -1])*2
    ax.plot(jnp.ones_like(theta)*d/2, jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[0], lw=.5)
    ax.plot(-d/2*jnp.ones_like(theta), jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[1], lw=.5)
    # Turn off the pane backgrounds (the greyish background)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Remove ticks from each axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    # Set the background color (optional)
    fig.patch.set_facecolor('white')
    # labels
    ax.text(-.25, -ylim[1], -zlim[1]-.5, 'x')  # X-axis label
    ax.text(xlim[1]+.25, -.25, -zlim[1], 'y')  # Y-axis label
    ax.text(xlim[1], ylim[1], zlim[1]+.2, 'z')  # Z-axis label
    # Set axis limits to create zoom effect
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    return ax


# figsize = (12*cm, 4*cm)
# fig = plt.figure(figsize=figsize)
# idxs_to_plot = [0, 2, -5]
# for cnt, idx in enumerate(idxs_to_plot):
#     data, y = datasets[idx], ys[idx]
#     magicnum = int(f'1{len(idxs_to_plot)}{cnt+1}')
#     newax = add_3d_scatter_with_custom_axes(fig, data, y, magicnum)
#     newax.set_title(f"$N_{{rots}} = {len(data)//2}$")
# plt.tight_layout()
# plt.savefig('images/fig3_panelA.pdf')
# plt.show()
# %% PANEL B
fig, ax = plt.subplots(figsize=(5*cm, 8*cm))
ax.plot(Ns, jnp.log(jnp.array(errors)).flatten(), 'o-', label=r'$L=1$')
ax.set_xlabel(r'$N_{{rot}}$')
ax.set_ylabel(r'$\log(\varepsilon)$', ha='left', rotation=0, y=1, labelpad=0)
ax.set_xticks([Ns[0], 25, 50])
textstr = '\n'.join((r'$L=1$', '$\Delta=1$'))
ax.text(
    0.5, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey')
)
# ax.legend()
plt.tight_layout()
plt.savefig('images/fig3_panelB.pdf')

plt.show()
# %% PANEL C
figsize = (12*cm, 4*cm)

def theory_spectrum(d: float, L: float=1.):
    def _inner(w):
        A = (d/L)**2
        return (1-A/2)*jnp.exp(-(w/2)**2) + (A/2) * jnp.exp(-((w-jnp.pi)/2)**2)
    return _inner

# fig, axs = plt.subplots(nrows=1, ncols=len(idxs_to_plot), figsize=figsize, layout='constrained')
# for ax, k in zip(axs, kernels):
#     n = len(k) // 2 + 1
#     ws = jnp.arange(n)
#     isp = 1/jnp.abs(orthorfft(k[0]))
#     ax.plot(jnp.arange(n), isp, color=colors[0])
#     ax.fill_between(jnp.arange(n), 0, isp, color=colors[0], alpha=.5)
#     ax.fill_between(ws, 0, isp[-1], color='grey', linestyle='--', alpha=.5)
#     ax.set_xlim(0, None)
#     ax.set_ylim(0, None)
# fig.supxlabel('Frequency', )
# fig.supylabel('$\lambda^{-1}$')
# plt.savefig('images/fig3_panelC.pdf')
# plt.show()


# %% TEST: PANELS A-C TOGETHER
# figsize = (12*cm, 8*cm)
# fig = plt.figure(figsize=figsize)
# idxs_to_plot = [0, 1, 4]
# for cnt, idx in enumerate(idxs_to_plot):
#     # row 1
#     data, y = datasets[idx], ys[idx]
#     magicnum = int(f'2{len(idxs_to_plot)}{cnt+1}')
#     r1_ax = add_3d_scatter_with_custom_axes(fig, data, y, magicnum)
#     r1_ax.set_title(f"$N_{{rots}} = {len(data)//2}$")
#     # row 2
#     k = kernels[idx]
#     ws = jnp.arange(len(k) // 2 + 1)
#     isp = 1/jnp.abs(orthorfft(k[0]) + 1e-5)
#     isp /= jnp.max(isp)
#     magicnum = int(f'2{len(idxs_to_plot)}{len(idxs_to_plot)+cnt+1}')
#     r2_ax = fig.add_subplot(magicnum)
#     r2_ax.plot(ws, isp, color=colors[0])
#     r2_ax.fill_between(ws, 0, isp, color=colors[0], alpha=.5)
#     r2_ax.fill_between(ws, 0, isp[-1], color='grey', linestyle='--', alpha=.5)
#     r2_ax.set_xlim(0, None)
#     r2_ax.set_ylim(0, 1.5)
#     r2_ax.set_xticks([0, len(k)//2], ['0', 'N'])
#     # r2_ax.set_ylabel('$\lambda^{-1}$', fontsize=20)
#     # r2_ax.set_yticks([])
#     # if cnt == 1:
#     #     r2_ax.set_yticks([0, 500, 1000], ['0', '5e1', '1e3'])
#     # if cnt == 2:
#     #     r2_ax.set_yticks([0, 1e4, 2e4], ['0', '1e4', '2e4'])
# fig.supxlabel('Frequency', y=.075)
# fig.supylabel('$\lambda^{-1}$', x=.05, y=.35)
# plt.subplots_adjust(wspace=0., hspace=0.)
# # plt.tight_layout()
# plt.savefig('images/fig3_AC.pdf')
# plt.show()
# %% AC PANELS TOGETHER, BUT NICE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


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
    ax.set_title(f"$N_{{rot}} = {len(xdata)//2}$")
    # data
    ax.scatter(xdata[1::2, 2], xdata[1::2, 1], xdata[1::2, 0], c=colors[1], marker='x', depthshade=False)
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
    ws = jnp.arange(len(k) // 2 + 1)
    isp = 1/jnp.abs(orthorfft(k[0]) + 1e-5)
    isp /= jnp.max(isp)
    ax.plot(ws, isp, color=colors[0])
    ax.fill_between(ws, 0, isp, color=colors[0], alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='grey', linestyle='--', alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='None', edgecolor='grey', linestyle='--', alpha=.5, hatch='/////')
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, len(k)//2], ['0', 'N'])
    # add text indicating last eigenvalue
    if axidx==2:
        axins = ax.inset_axes(
            [0.6, 0.6, 0.6, 0.6],
            xlim=(len(k)//2-3, len(k)//2+.2), ylim=(0, isp[-4]),
            xticklabels=[], yticklabels=[])
        axins.plot(ws[-4:], isp[-4:], color=colors[0])
        axins.fill_between(ws[-4:], 0, isp[-4:], color=colors[0], alpha=.5)
        axins.fill_between(ws[-4:], 0, isp[-1], color='grey', linestyle='--', alpha=.5)
        axins.fill_between(ws[-4:], 0, isp[-1], color='None', edgecolor='grey', linestyle='--', alpha=.5, hatch='/////')
        axins.text((len(k)//2)*1.01, isp[-1], r'$\lambda^{-1}_N$')
        axins.scatter(len(k)//2, isp[-1], c='grey', s=3, zorder=10)
        ax.indicate_inset_zoom(axins, edgecolor="black")
    else:
        ax.text((len(k)//2)*(1.1), isp[-1], r'$\lambda^{-1}_N$')
        ax.scatter(len(k)//2, isp[-1], c='grey', s=3, zorder=10)

fig.supylabel(r'norm$(\lambda^{-1})$', x=0.1, y=.3, ha='center')
fig.supxlabel('Frequency', ha='center', y=0.09)
plt.tight_layout()
plt.savefig('images/fig3_panelsAC.pdf')
plt.show()


# %% PANELS D, E, F: data
N = 6
L = .85
r = 1.
ds = jnp.array([0.] + list(jnp.logspace(-2, 1, 20)))

get_k = lambda data: jnp.exp(-L**2 * jnp.sum((data-data[:, None])**2, axis=-1))
datasets = [get_data(N, d, r) for d in ds]
ys = jnp.array([-1., 1.]*N)[:, None]
kernels = [get_k(data) for data in datasets]

errors = []
for k in kernels:
    k_train_train = jnp.delete(jnp.delete(k, N//2, 1), N//2, 0)
    k_train_test = jnp.delete(k[:, N//2:N//2+1], N//2, 0)
    k_test_test = k[N//2:N//2+1, N//2:N//2+1]
    y_train = jnp.delete(ys, N//2, axis=0)
    ymean, yvar = kreg(k_train_train, k_train_test, k_test_test, y_train)
    errors.append(jnp.abs(ys[N//2] - ymean))

errors = jnp.array(errors).flatten()
# plt.plot(jnp.log(ds), jnp.log(errors))
# plt.show()

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
    ax.scatter(xdata[1::2, 2], xdata[1::2, 1], xdata[1::2, 0], c=colors[1], marker='x', depthshade=False)
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
    ws = jnp.arange(len(k) // 2 + 1)
    isp = 1/jnp.abs(orthorfft(k[0]) + 1e-5)
    isp /= jnp.max(isp)
    ax.plot(ws, isp, color=colors[0])
    ax.fill_between(ws, 0, isp, color=colors[0], alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='grey', linestyle='--', alpha=.5)
    ax.fill_between(ws, 0, isp[-1], color='None', edgecolor='grey', linestyle='--', alpha=.5, hatch='/////')
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, len(k)//2], ['0', 'N'])
    # add text indicating last eigenvalue
    ax.text((len(k)//2)*(1.1), isp[-1], r'$\lambda^{-1}_N$')
    ax.scatter(len(k)//2, isp[-1], c='grey', s=3, zorder=10)

fig.supylabel(r'$\lambda^{-1}$', x=0.1, y=.3, ha='center')
fig.supxlabel('Frequency', ha='center', y=0.09)
plt.tight_layout()
plt.savefig('images/fig3_panelsDF.pdf')
plt.show()


# %% PANEL E
figsize = (5*cm, 8*cm)
fig, ax = plt.subplots(figsize=figsize)
ax.plot(jnp.log(ds), jnp.log(jnp.array(errors)).flatten(), 'o-', label='$N=1, L=1$')
ax.set_xlabel(r'$\log\Delta$')
ax.set_ylabel(r'$\log(\varepsilon)$', ha='left', rotation=0, y=1, labelpad=0)
textstr = '\n'.join((r'$L=1$', f'$N_{{rots}}={N}$'))
ax.text(
    0.5, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey')
)
# ax.legend()
plt.tight_layout()
plt.savefig('images/fig3_panelE.pdf')
plt.show()
