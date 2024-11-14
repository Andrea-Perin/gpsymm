# %% High-d case, but this time with trained networks instead of NTK
# import pdb
# pdb.set_trace()

import numpy as np
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
from gp_utils import make_circulant, circulant_error
Ensemble = List[Tuple[Array, Array] | Tuple[()]]

# %% Parameters
SEED = 124
RNG = jr.PRNGKey(SEED)
N_ROTATIONS = [4, 8, 16, 32, 64]
N_PAIRS = 250
N_IMGS = 60_000
IN_SHAPE = 784
HIDDEN_DIM = 512
N_THEORY_VALS = len(['sp_err', 'lambda_n', 'lambda_avg', 'deltasq', 'avg_angle'])
W_std = 1.
b_std = 1.
N_EPOCHS = 15_000
REG = 1e-5
out_path = Path('results/highd_trained_weirdinit')
out_path.mkdir(parents=True, exist_ok=True)


img_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
lab_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
images = normalize_mnist(load_images(img_path=img_path))
labels = load_labels(lab_path=lab_path)
make_orbit = kronmap(three_shear_rotate, 2)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
in_shape = images[0].flatten().shape
# network and NTK
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(HIDDEN_DIM, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
    nt.stax.Dense(1, W_std=W_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)


def kaiming_uniform(in_, shape, key):
    amax = 1/jnp.sqrt(in_)
    return jr.uniform(key, shape=shape, minval=-amax, maxval=amax)


def init_params(key, uniform: bool = False):
    key, ens_key = jr.split(key)
    ens_key = jr.split(ens_key, N_PAIRS)
    out_shape, params = jax.vmap(init_fn, in_axes=(0, None))(ens_key, in_shape)
    if uniform:
        # if here, initialize like PyTorch does. SUPER HACKY
        nlayers = len(params)
        new_params = []
        for l in range(nlayers):
            key, lk = jr.split(key)
            wk, bk = jr.split(lk)
            if params[l] == ():  # activation layer
                new_params.append( () )
            else:  # linear layer
                new_params.append(
                    (
                        kaiming_uniform(params[l][0].shape[1], params[l][0].shape, wk),
                        kaiming_uniform(params[l][1].shape[-1], params[l][1].shape, bk)
                    )
                )
        params = new_params
    return key, params


# %% Optimizer
schedule = optax.schedules.warmup_exponential_decay_schedule(
    init_value=1e-3,
    peak_value=5e-1,
    warmup_steps=1_000,
    transition_steps=100,
    decay_rate=.5
)
optim = optax.novograd(learning_rate=schedule)


# %%
def loss(params: Ensemble, x: Float[Array, 'pair 2*angles 28*28'], y: Float[Array, 'pair 2*angles']) -> Scalar:
    yhat = jax.vmap(apply_fn)(params, x)
    mses = ein.reduce((y[..., None]-yhat)**2, 'pair angle d -> pair', 'mean')
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
        updates, opt_state = optim.update(grads, opt_state, params)
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
    collision_rate: float = 1.,
) -> Float[Array, 'pair (angle digit) (width height)']:
    """BEWARE: This thing draws on and on until it gets n_pairs that are good"""
    n_pairs_actual = int(n_pairs * (1+collision_rate))
    angles = jnp.linspace(0, 2*jnp.pi, n_rotations, endpoint=False)
    num_produced = 0
    while num_produced < n_pairs:
        key, key_ab = jr.split(key)
        idxs_A, idxs_B = jr.randint(key_ab, minval=0, maxval=N_IMGS, shape=(2, n_pairs_actual,))
        # remove same-digit pairs
        labels_A, labels_B = labels[idxs_A], labels[idxs_B]
        collision_mask = (labels_A == labels_B)
        idxs_A, idxs_B = idxs_A[~collision_mask][:n_pairs], idxs_B[~collision_mask][:n_pairs]
        num_produced = len(idxs_A)
    images_A, images_B = images[idxs_A], images[idxs_B]
    orbits_A = make_orbit(images_A, angles)
    orbits_B = make_orbit(images_B, angles + jnp.pi/n_rotations)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * angle width height')
    return ein.rearrange(data, 'pair digit angle width height -> pair (angle digit) (width height)')


@jax.jit
def get_theory_values(data: Float[Array, 'pair (angle digit) (width height)']) -> Float[Array, 'pair N_THEORY_VALS']:
    # compute the spectral error, lambda_N, lambda_avg, deltasq, and the angle
    kernels = jax.vmap(kernel_fn)(data, data).ntk
    ckernels = jax.vmap(make_circulant)(kernels)
    isp = 1/jnp.abs(jax.vmap(orthofft)(ckernels[:, 0]) + REG*jnp.sqrt(ckernels.shape[-1]))
    lam_n = isp[:, isp.shape[1]//2]
    lam_avg = ein.reduce(isp, 'pair spectrum -> pair', 'mean')
    sp_err = jax.vmap(circulant_error)(ckernels)
    # deltasq
    rearranged = ein.rearrange(data, 'pair (angle digit) wh -> digit pair angle wh', digit=2)
    projs = ein.reduce(rearranged, 'digit pair angle wh -> digit pair wh', 'mean')
    deltasq = ein.einsum((projs[0] - projs[1])**2, 'pair wh -> pair')
    # avg angle
    avg_angle = ckernels[:, 0, 2]
    return ein.pack(
        (sp_err, lam_n, lam_avg, deltasq, avg_angle),
        'pair *')[0]


theory_values = np.empty( (len(N_ROTATIONS), N_PAIRS, N_THEORY_VALS) )
preds = np.empty( (len(N_ROTATIONS), N_PAIRS, 2) )
losses = np.empty( (len(N_ROTATIONS), N_EPOCHS, 2) )
for angle_idx, n_angles in enumerate(tqdm(N_ROTATIONS)):
    data = get_data(key=RNG, n_rotations=n_angles, n_pairs=N_PAIRS)
    targets = jnp.array(([+1., -1.]*n_angles))
    targets = ein.repeat(targets, 'angle -> pair angle', pair=N_PAIRS)
    # compute the spectral error, lambda_N, lambda_avg, deltasq, and the angle
    theory_values[angle_idx] = get_theory_values(data)
    # train on positive
    RNG, params_p = init_params(RNG, uniform=True)
    params_p, losses_p = train(params=params_p, x=data[:, 1:], y=targets[:, 1:], optim=optim, epochs=N_EPOCHS)
    preds_p = jax.vmap(apply_fn)(params_p, data[:, 0])
    preds[angle_idx, :, :1] = preds_p
    losses[angle_idx, :, 0] = losses_p
    del params_p
    # train on negative
    RNG, params_m = init_params(RNG, uniform=True)
    params_m, losses_m = train(params=params_m, x=data[:, :-1], y=targets[:, :-1], optim=optim, epochs=N_EPOCHS)
    preds_m = jax.vmap(apply_fn)(params_m, data[:, -1])
    preds[angle_idx, :, 1:] = preds_m
    losses[angle_idx, :, 1] = losses_m
    del params_m


# save data
np.save(out_path / 'theory_values', theory_values)
np.save(out_path / 'predictions', preds)
np.save(out_path / 'losses', losses)
print(f"Saved results in {out_path}")
