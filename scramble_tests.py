# %% Investigations on taking permutations of digits and checking what happens to the NTK
import numpy as np
from scipy.interpolate import griddata
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, Scalar, PyTree, Int, PRNGKeyArray, UInt8
from typing import Tuple
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft

from mnist_utils import load_images, load_labels, normalize_mnist
from data_utils import three_shear_rotate, xshift_img, kronmap
from gp_utils import kreg, circulant_error, make_circulant, extract_components
from plot_utils import cm, cloudplot

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('myplots.mlpstyle')


# %% Parameters
SEED = 123354
RNG = jr.PRNGKey(SEED)
SHIFTS = jnp.arange(10, dtype=int)  # we will shift by 1, 10 times
N_PAIRS = 10  # keep this small; the computations are heavy
REG = 1e-5
N_PCS = 3
RANDOM_DATA = False


# %% A quenched pixel scrambling + shifting
RNG, pk = jr.split(RNG)
perm = jr.permutation(key=pk, x=28*28)
invperm = jnp.argsort(perm)

def weird_shift(img, shift):
    """Scrambles pixels in an image, flattens it, shifts it by `shift` pixels,
    then unscrambles it and reshapes it."""
    fimg = ein.rearrange(img, 'w h -> (w h)')
    pfimg = jnp.roll(fimg[perm], shift)
    return ein.rearrange(pfimg[invperm], '(w h) -> w h', w=28)


# %% Load data
img_path = './data/mnist/raw/train-images-idx3-ubyte.gz'
lab_path = './data/mnist/raw/train-labels-idx1-ubyte.gz'
images = normalize_mnist(load_images(img_path=img_path))
labels = load_labels(lab_path=lab_path)
make_orbit = kronmap(weird_shift, 2)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
# network and ntk; we will study it for a CNN
w_std, b_std = 1., 1.
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Conv(128, (4, 4), padding='CIRCULAR'),
    nt.stax.Relu(),
    # nt.stax.GlobalAvgPool(),
    nt.stax.Flatten(),
    nt.stax.Dense(1)
)
kernel_fn = jax.jit(kernel_fn)


# %% Examples of what the weird shift function does
plt.imshow(weird_shift(images[0], 0))
plt.show()
plt.imshow(weird_shift(images[0], 1))
plt.show()
plt.imshow(weird_shift(images[0], 2))
plt.show()
plt.imshow(weird_shift(images[0], 3))
plt.show()
plt.imshow(weird_shift(images[0], 28*28))
plt.show()


# %% Creating data and inspecting the related kernels
def get_data(
    key: PRNGKeyArray,
    shifts: Int[Array, 'n'],
    n_pairs: int = N_PAIRS,
    collision_rate: float = .2,
) -> Float[Array, 'pair (shift digit) (width height)']:
    n_pairs_actual = int(n_pairs * (1+collision_rate))
    key_a, key_b = jr.split(key)
    idxs_A, idxs_B = jr.randint(key, minval=0, maxval=len(images), shape=(2, n_pairs_actual,))
    # remove same-digit pairs
    labels_A, labels_B = labels[idxs_A], labels[idxs_B]
    collision_mask = (labels_A == labels_B)
    idxs_A, idxs_B = idxs_A[~collision_mask][:n_pairs], idxs_B[~collision_mask][:n_pairs]
    #
    images_A, images_B = images[idxs_A], images[idxs_B]
    orbits_A = make_orbit(images_A, shifts)
    orbits_B = make_orbit(images_B, shifts)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * shift width height')
    return ein.rearrange(data, 'pair digit shift width height -> pair (shift digit) (width height)')


# RANDOM_DATA: use if you want to test something like an MLP
# For structured things, like a CNN, using scrambled shifts of a proper digit is
# more insightful
if RANDOM_DATA:
    data = jr.normal(RNG, shape=(N_PAIRS, 2*len(SHIFTS), 28*28))
else:
    data = get_data(key=RNG, shifts=SHIFTS, n_pairs=N_PAIRS)
    data = ein.rearrange(data, 'pair shift_digit (width height) -> pair shift_digit width height 1', width=28)

kernels = jax.vmap(kernel_fn)(data, data).ntk


# %% Show kernels
# first, for each pair, the kernel of a single class
for i in range(N_PAIRS):
    plt.close()
    plt.figure()
    plt.imshow( kernels[i, ::2, ::2])
    plt.colorbar()
    plt.show()
# then, for the first pair, both classes
plt.close()
plt.figure()
plt.imshow( kernels[0])
plt.colorbar()
plt.show()
