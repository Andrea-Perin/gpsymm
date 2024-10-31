# %% Contains the spectrum sketch figure
from pathlib import Path
import jax
from jax import numpy as jnp
from jaxtyping import Float, Array
import einops as ein

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# plt.style.use('./myplots.mplstyle')

from plot_utils import cm, add_spines, clean_3d_ax

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bcmap = mpl.colors.ListedColormap([colors[0], colors[1]])
out_path = Path('images/spectrum_sketch')
out_path.mkdir(parents=True, exist_ok=True)


# first part is just a 3D plot like in figure
def get_data(N: int, d: float, r: float=1.,) -> Float[Array, '2N 3']:
    angles_p = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
    angles_m = angles_p + jnp.pi/N
    xs_p, _ = ein.pack( (r*jnp.cos(angles_p), r*jnp.sin(angles_p), jnp.ones(N)*d/2), 'd *' )
    xs_m, _ = ein.pack( (r*jnp.cos(angles_m), r*jnp.sin(angles_m), -d/2*jnp.ones(N)), 'd *' )
    data, ps = ein.pack((xs_p, xs_m), '* d')
    return ein.rearrange(data, '(s n) d -> (n s) d', n=N)


L = 1.
N = 8
D = 1.
R = 1.
data = get_data(N=N, d=D, r=R)
ys = jnp.array([1., -1.]*(len(data)//2))[:, None]
distsq = jnp.sum((data - data[:, None])**2, axis=-1)
kernel = jnp.exp(-L**2 * distsq)

# %% plot the 3d setup
fig = plt.figure(figsize=(6.25*cm, 6*cm))
ax = fig.add_subplot(111, projection='3d')
ax = clean_3d_ax(ax)
ax.scatter(data[1::2, 2], data[1::2, 1], data[1::2, 0], c=colors[1], marker='o', depthshade=False)
ax.scatter(data[::2, 2], data[::2, 1], data[::2, 0], c=colors[0], marker='o', depthshade=False)
# add fake circle
theta = jnp.linspace(0, 2 * jnp.pi, 201)
ax.plot(jnp.ones_like(theta)*D/2, jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[0], lw=.5)
ax.plot(-D/2*jnp.ones_like(theta), jnp.sin(theta), jnp.cos(theta), linestyle='--', c=colors[1], lw=.5)
ax.text2D(0.25, 0.05, "x", transform=ax.transAxes)
ax.text2D(0.85, 0.15, "y", transform=ax.transAxes)
ax.text2D(0.95, 0.85, "z", transform=ax.transAxes)
plt.tight_layout()
plt.savefig(out_path / 'panelA.pdf')
plt.show()
# %% Plot the distance matrix and the kernel
fig, ax = plt.subplots(
    figsize=(4*cm, 6*cm),
    nrows=2, ncols=1,
    sharex=True, sharey=True,
)

ax[0].set_axis_off()
ax[0].imshow(distsq)
ax[0].set_title(r"$d^2(x_i, x_j)$", fontsize=10)

ax[1].set_axis_off()
ax[1].imshow(kernel)
ax[1].set_title(r"$k_{RBF}(x_i, x_j)$", fontsize=10)
# highlight one row of the kernel matrix
row = N
rect1 = Rectangle( (-.5, row-.5), width=2*N, height=1, linewidth=1, edgecolor=colors[0], facecolor='none')
ax[1].add_patch(rect1)
ax[1].text(-1, row, r'$k(x_N, \vec{x}_\mathcal{D})$', horizontalalignment='right', verticalalignment='center', rotation=90)
plt.subplots_adjust(wspace=0, hspace=.4)
plt.savefig(out_path / 'panelB.pdf')
plt.show()
# %% Plot the kernel row
krow = kernel[row]
spectrum = 1/jnp.abs(jnp.fft.rfft(krow))
spectrum /= spectrum.max()


fig, ax = plt.subplots(
    figsize=(6.25*cm, 6*cm),
    nrows=2, ncols=1,

)

# compute the bounding lines for the kernel
angle_coarse = jnp.linspace(0, 2*jnp.pi, 2*N, endpoint=False)
angle_fine = jnp.linspace(0, 2*jnp.pi, 360, endpoint=False)
xy, _ = ein.pack((jnp.cos(angle_fine), jnp.sin(angle_fine)), 'd *')
distsq_circle = jnp.sum((xy[180]-xy)**2, axis=-1)
lower_bound = jnp.exp(-L**2 * distsq_circle)
upper_bound = jnp.exp(-L**2 * (distsq_circle + D**2))

ax[0].plot(angle_coarse, krow, '.-', label='$k_{RBF}$')
ax[0].plot(angle_fine, upper_bound, ls='--', label='Opp.')
ax[0].plot(angle_fine, lower_bound, ls='--', label='Same')
ax[0].set_yticks([0, 1])
ax[0].set_xticks(angle_coarse[::4])
ax[0].set_xticklabels(jnp.arange(-N, N)[::4])
ax[0].set_xlabel('$j-N$')
ax[0].set_ylabel('$k_{RBF}$')
ax[0].legend(loc='upper right', handlelength=1., handletextpad=.5, bbox_to_anchor=(1.1, 1))

freqs = jnp.arange(N+1)
ax[1].set_ylabel('$\lambda^{-1}$')
ax[1].set_xlabel('Frequency index')
ax[1].set_xticks([0, N], ['1', 'N'])
ax[1].plot(freqs, spectrum)
ax[1].fill_between(freqs, 0, spectrum, color=colors[0], alpha=.5)
ax[1].hlines(spectrum[-1], 0, freqs[-1], color='grey', ls='--')
ax[1].fill_between(freqs, 0, spectrum[-1], color='grey', linestyle='--', alpha=.5)
ax[1].fill_between(freqs, 0, spectrum[-1], color='None', edgecolor='grey', linestyle='--', alpha=.5, hatch='/////')
ax[1].set_xlim((0, None))
ax[1].set_ylim((0, 1.3))
ax[1].text(N, spectrum[-1]*1.5, r'$\lambda^{-1}_N$')
ax[1].scatter(N, spectrum[-1], color='grey', s=4)
plt.tight_layout(pad=0)
plt.savefig(out_path / 'panelC.pdf')
plt.show()
