# %%
import jax
from jax import numpy as jnp
import einops as ein
from pathlib import Path
from mnist_utils import load_images, load_labels, normalize_mnist
from data_utils import three_shear_rotate, kronmap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('./myplots.mlpstyle')
from plot_utils import cm, get_size


labpath = Path('./data/MNIST/raw/train-labels-idx1-ubyte.gz')
imgpath = Path('./data/MNIST/raw/train-images-idx3-ubyte.gz')
images = normalize_mnist(load_images(img_path=imgpath))
labels = load_labels(lab_path=labpath)


# %% PANEL A: sketched MNIST orbits
N_angles = 8
label_a, label_b = 4, 7
angles = jnp.linspace(0, 1, N_angles, endpoint=False) * jnp.pi * 2

digit_a = images[labels == label_a]
digit_b = images[labels == label_b]
make_orbit_grid = kronmap(three_shear_rotate, 2)
digit_a_orbits = make_orbit_grid(digit_a[:1], angles)
digit_b_orbits = make_orbit_grid(digit_b[:1], angles)
# %% Project an orbit digit in 3d
N_PCS = 3
data, ps = ein.pack( (digit_a_orbits[0], digit_b_orbits[0]), '* n w h' )
data = ein.rearrange(data, 'd n w h -> (n d) (w h)')
u, s, vh = jnp.linalg.svd(data,  full_matrices=True)
pcs = ein.einsum(u[:, :N_PCS], s[:N_PCS], 'i j, j -> i j')


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcs[::2, 0], pcs[::2, 1], pcs[::2, 2], label=f'Digit: {label_a}')
ax.scatter(pcs[1::2, 0], pcs[1::2, 1], pcs[1::2, 2], label=f'Digit: {label_b}')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('3D Projection of Digit Orbit')
plt.legend()
plt.tight_layout()
plt.show()


# vh contains the principal components' directions
pcs_for_plotting = ein.rearrange(vh, 'pc (w h) -> pc w h', w=28, h=28)

fig = plt.figure(figsize=(9, 9))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(3, 3),
                 axes_pad=0.1,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.05,
                 )
vmin = jnp.min(pcs_for_plotting[:len(grid)])
vmax = jnp.max(pcs_for_plotting[:len(grid)])
# Plot each image
for idx, (ax, pc) in enumerate(zip(grid, pcs_for_plotting)):
    im = ax.imshow(pc, vmin=vmin, vmax=vmax)
    ax.set_title(f"PC {idx+1}")
    ax.axis('off')

grid.cbar_axes[0].colorbar(im)
plt.suptitle("Principal Components", fontsize=16)
plt.tight_layout()
plt.show()



# %% ACTUAL PLOTS
Ns = [4, 8, 16, 32]
