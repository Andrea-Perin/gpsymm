# %% Plotting principal components
from jax import numpy as jnp
import einops as ein
from pathlib import Path

from utils.mnist_utils import load_images, load_labels, normalize_mnist
from utils.data_utils import make_rotation_orbit
from utils.plot_utils import cm, add_spines

import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
plt.style.use('myplots.mlpstyle')


def main():
    # Parameters
    N_ROTATIONS = 8

    out_path = Path(f'images/pca_plot')
    out_path.mkdir(parents=True, exist_ok=True)

    img_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
    lab_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
    images = load_images(img_path=img_path)
    labels = load_labels(lab_path=lab_path)

    # %% PANEL A to start
    label_a, label_b = 4, 7
    digit_selector = 43  # this is just a good looking pair of digits
    angles = jnp.linspace(0, 360, angles_panel_a:=360, endpoint=False)
    digit_a = images[labels == label_a][digit_selector:digit_selector+1]
    digit_b = images[labels == label_b][digit_selector:digit_selector+1]

    digit_a_orbits = make_rotation_orbit(digit_a, angles)
    digit_b_orbits = make_rotation_orbit(digit_b, angles)
    digit_a_orbits = normalize_mnist(digit_a_orbits[0])
    digit_b_orbits = normalize_mnist(digit_b_orbits[0])

    data, ps = ein.pack( (digit_a_orbits, digit_b_orbits), '* n w h' )
    data = ein.rearrange(data, 'd n w h -> (d n) (w h)')
    u, s, vh = jnp.linalg.svd(data, full_matrices=True)
    comps = jnp.array([0, 1, 2])
    pcs = ein.einsum(u[:, comps], s[comps], 'i j, j -> i j')

    # Extract points for both orbits
    idxs = jnp.arange(angles_panel_a + N_ROTATIONS, step=angles_panel_a//N_ROTATIONS)
    pcs_a = pcs[:angles_panel_a]
    pcs_b = pcs[angles_panel_a:]
    pts_a = jnp.take(pcs_a, idxs, axis=0, mode='wrap')
    pts_b = jnp.take(pcs_b, idxs, axis=0, mode='wrap')

    # Main scatter plot
    fig = plt.figure(figsize=(5.75*cm, 5*cm))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position((0, 0, 1, 1))
    ax.plot(pts_a[:, 0], pts_b[:, 1], pts_a[:, 2], 'o--', lw=.3, markersize=1.5, label=label_a)
    ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], 'o--', lw=.3, markersize=1.5, label=label_b)
    # show PCA basis elements
    vmax = max(float(vh[:3].max()), float(-vh[:3].min()))
    vmin = -vmax
    inset_pc1 = ax.inset_axes((.25, 0.1, 0.15, 0.15), transform=ax.transAxes)
    inset_pc1.set_title("PC1", fontsize=10, y= -.5, va='center', transform=inset_pc1.transAxes)
    add_spines(inset_pc1)
    inset_pc1.imshow(-vh[0].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)  # adjust index as needed

    inset_pc2 = ax.inset_axes((.75, 0.1, 0.15, 0.15))
    inset_pc2.set_title("PC2", fontsize=10, y=-.5, va='center', transform=inset_pc2.transAxes)
    add_spines(inset_pc2)
    inset_pc2.imshow(vh[1].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)  # adjust index as needed

    inset_pc3 = ax.inset_axes((.9, 0.5, 0.15, 0.15))
    inset_pc3.set_title("PC3", fontsize=10, y=-.5, va='center', transform=inset_pc3.transAxes,
        path_effects=[withStroke(linewidth=3, foreground='w')])
    add_spines(inset_pc3)
    inset_pc3.imshow(vh[2].reshape(28, 28), cmap='RdBu', vmin=vmin, vmax=vmax)  # adjust index as needed
    # Show digits
    inset1 = ax.inset_axes((0.05, 0.75, 0.15, 0.15))
    inset1.set_axis_off()
    inset1.imshow(digit_a[0], cmap='gray')
    inset2 = ax.inset_axes((0.70, 0.75, 0.15, 0.15))
    inset2.set_axis_off()
    inset2.imshow(digit_b[0], cmap='gray')
    # Plot images in insets
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=0)
    plt.savefig(out_path / f'pca_plot_{N_ROTATIONS}.pdf', bbox_inches='tight') #, pad_inches=0)
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
