# %% High-d case, but this time with trained networks instead of NTK
import numpy as np
from scipy.interpolate import griddata
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, Scalar, PyTree, Int, PRNGKeyArray, UInt8
import equinox as eqx
import optax
from typing import Tuple, List
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft
from pathlib import Path

from mnist_utils import load_images, load_labels, normalize_mnist
from data_utils import three_shear_rotate, xshift_img, kronmap
from gp_utils import kreg, circulant_error, make_circulant, extract_components
from plot_utils import cm, cloudplot, add_spines

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patheffects import withStroke
plt.style.use('myplots.mlpstyle')
plt.set_cmap('viridis')
Ensemble = List[Tuple[Array, Array] | Tuple[()]]

# %% Parameters
SEED = 124
RNG = jr.PRNGKey(SEED)
N_ROTATIONS = [4, 8, 16, 32, 64]
N_PAIRS = 1000

N_IMGS = 60_000
IN_SHAPE = 784
HIDDEN_DIM = 512
N_EPOCHS = 5_000
REG = 1e-5
N_PCS = 3
out_path = Path('images/highd_trained')
out_path.mkdir(parents=True, exist_ok=True)


img_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
lab_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
images = normalize_mnist(load_images(img_path=img_path))
labels = load_labels(lab_path=lab_path)
make_orbit = kronmap(three_shear_rotate, 2)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
in_shape = images[0].flatten().shape
# network and NTK
W_std, b_std = 1., 1.
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(HIDDEN_DIM, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
    nt.stax.Dense(1, W_std=W_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)

def init_params(key):
    key, ens_key = jr.split(key)
    ens_key = jr.split(ens_key, N_PAIRS)
    out_shape, params = jax.vmap(init_fn, in_axes=(0, None))(ens_key, in_shape)
    return key, params


# %% Optimizer
schedule = optax.schedules.warmup_exponential_decay_schedule(
    init_value=1e-3,
    peak_value=5e-1,
    warmup_steps=1_000,
    transition_steps=100,
    decay_rate=.5
)
optim = optax.sgd(learning_rate=schedule)





# %%
def loss(params: Ensemble, x: Float[Array, 'pair 2*angles 28*28'], y: Float[Array, 'pair 2*angles']) -> Scalar:
    yhat = jax.vmap(jax.vmap(apply_fn, in_axes=(None, 0)))(params, x)
    yhat = yhat.squeeze()
    mses = ein.reduce((y-yhat)**2, 'pair angle -> pair', 'mean')
    return jnp.sum(mses)


def train(
    params: Ensemble,
    x: Float[Array, 'pairs 2*angles 28*28'],
    y: Float[Array, 'pairs 2*angles'],
    optim: optax.GradientTransformation,
    epochs: int,
) -> Tuple[Ensemble, List[float]]:

    opt_state = optim.init(params)

    @eqx.filter_jit
    def make_step(
        params: Ensemble,
        opt_state: PyTree,
        x: Float[Array, "2*angles 28*28"],
        y: Int[Array, " 2*angles"],
    ):
        loss_value, grads = jax.value_and_grad(loss)(params, x, y)
        updates, opt_state = optim.update(grads, opt_state, params )
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    for epoch in range(epochs):
        params, opt_state, train_loss = make_step(params, opt_state, x, y)
        losses.append(train_loss)

    return params, losses


# select pairs from original mnist
def get_data(
    key: PRNGKeyArray,
    n_rotations: int,
    n_pairs: int = N_PAIRS,
    collision_rate: float = .2,
) -> Float[Array, 'pair (angle digit) (width height)']:
    n_pairs_actual = int(n_pairs * (1+collision_rate))
    key_a, key_b = jr.split(key)
    angles = jnp.linspace(0, 2*jnp.pi, n_rotations, endpoint=False)
    idxs_A, idxs_B = jr.randint(key, minval=0, maxval=N_IMGS, shape=(2, n_pairs_actual,))
    # remove same-digit pairs
    labels_A, labels_B = labels[idxs_A], labels[idxs_B]
    collision_mask = (labels_A == labels_B)
    idxs_A, idxs_B = idxs_A[~collision_mask][:n_pairs], idxs_B[~collision_mask][:n_pairs]
    #
    images_A, images_B = images[idxs_A], images[idxs_B]
    orbits_A = make_orbit(images_A, angles)
    orbits_B = make_orbit(images_B, angles + jnp.pi/n_rotations)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * angle width height')
    return ein.rearrange(data, 'pair digit angle width height -> pair (angle digit) (width height)')



preds = np.empty( (len(N_ROTATIONS), N_PAIRS, 2) )
for angle_idx, n_angles in enumerate(tqdm(N_ROTATIONS)):
    data = get_data(key=RNG, n_rotations=n_angles, n_pairs=N_PAIRS)
    targets = jnp.array(([+1., -1.]*n_angles))
    targets = ein.repeat(targets, 'angle -> pair angle', pair=N_PAIRS)
    # train on positive
    RNG, params_p = init_params(RNG)
    params_p, losses_p = train(params=params_p, x=data[:, 1:], y=targets[:, 1:], optim=optim, epochs=N_EPOCHS)
    preds_p = jax.vmap(apply_fn)(params_p, data[:, 0])
    preds[angle_idx, :, :1] = preds_p
    # train on negative
    RNG, params_m = init_params(RNG)
    params_m, losses_m = train(params=params_m, x=data[:, :-1], y=targets[:, :-1], optim=optim, epochs=N_EPOCHS)
    preds_m = jax.vmap(apply_fn)(params_m, data[:, -1])
    preds[angle_idx, :, 1:] = preds_m


np.save(out_path / 'predictions', preds)
# %%
