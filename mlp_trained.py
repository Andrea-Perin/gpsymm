# %% High-d case, but this time with trained networks instead of NTK
import numpy as np
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float, Scalar, PyTree, Int, PRNGKeyArray
import optax
import einops as ein
import neural_tangents as nt
from tqdm import tqdm
import functools as ft
from pathlib import Path
import argparse

from utils.conf import load_config
from utils.net_utils import kaiming_uniform_pytree
from utils.mnist_utils import load_images, load_labels, normalize_mnist
from utils.data_utils import make_rotation_orbit, get_idxs
from utils.gp_utils import make_circulant, circulant_error
Ensemble = PyTree

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MLP (trained) analysis')
parser.add_argument('--n-hidden', type=int, default=1,
                   help='number of hidden layers (default: 1)')
args = parser.parse_args()

# %% Parameters
cfg = load_config('config.toml')
SEED = cfg['params']['seed']
RNG = jr.PRNGKey(SEED)
N_ROTATIONS = cfg['params']['rotations']  # [4, 8, 16, 32, 64]
N_PAIRS = cfg['params']['n_pairs']
W_std = 1.
b_std = 1.
N_EPOCHS = cfg['params']['n_epochs']
BATCH_SIZE = cfg['params']['batch_size']

# %% Paths
res_path = Path(cfg['paths']['res_path']) / f'mlp_trained_{args.n_hidden}'
res_path.mkdir(parents=True, exist_ok=True)
img_path = Path(cfg['paths']['img_path'])
lab_path = Path(cfg['paths']['lab_path'])


N_IMGS = 60_000
HIDDEN_DIM = 512
N_THEORY_VALS = len(['sp_err', 'lambda_n', 'lambda_avg', 'deltasq', 'avg_angle'])
REG = 1e-5


images = load_images(img_path=img_path)
labels = load_labels(lab_path=lab_path)
orthofft = ft.partial(jnp.fft.fft, norm='ortho')
in_shape = images[0].flatten().shape
# network and NTK
# layer = nt.stax.serial(
#     nt.stax.Dense(512, W_std=W_std, b_std=b_std),
#     nt.stax.Relu(),
# )
# init_fn, apply_fn, kernel_fn = nt.stax.serial(
#     nt.stax.serial(*([layer] * args.n_hidden)),
#     nt.stax.Dense(1, W_std=W_std, b_std=b_std)
# )
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(512, W_std=W_std, b_std=b_std),
    nt.stax.Relu(),
    nt.stax.Dense(1, W_std=W_std, b_std=b_std)
)
kernel_fn = jax.jit(kernel_fn)


optim = optax.adam(learning_rate=1e-3)


# %%
def loss(params: PyTree, x: Float[Array, 'pair 2*angles 28*28'], y: Float[Array, 'pair 2*angles']) -> Scalar:
    yhat = jax.vmap(apply_fn)(params, x)
    mses = ein.reduce((y[..., None]-yhat)**2, 'pair angle d -> pair', 'mean')
    return jnp.sum(mses)


def train(
    params,
    train_x: Float[Array, 'ens 2*angle 28*28'],
    train_y: Float[Array, 'ens 2*angle'],
    optim,
    epochs,
    key
):
    opt_state = optim.init(params)

    @jax.jit
    def make_step(
        params,
        opt_state: PyTree,
        x: Float[Array, "batch 28*28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = jax.value_and_grad(loss)(params, x, y)
        updates, opt_state = optim.update( grads, opt_state, params )
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    keys = jr.split(key, num=epochs)
    num_points = train_x.shape[1]
    iters_per_epoch = (num_points // BATCH_SIZE + (num_points % BATCH_SIZE > 0))
    for epoch, key in enumerate(keys):
        # shuffle x and y - along second axis!
        key, pkey = jr.split(key)
        perm = jr.permutation(pkey, train_x.shape[1])
        train_x = train_x[:, perm]
        train_y = train_y[:, perm]
        for batch_idx in range(iters_per_epoch):
            params, opt_state, train_loss = make_step(
                params,
                opt_state,
                train_x[:, BATCH_SIZE*batch_idx: BATCH_SIZE*(batch_idx+1)],
                train_y[:, BATCH_SIZE*batch_idx: BATCH_SIZE*(batch_idx+1)],
            )
            losses.append(train_loss)
    return params, losses


# select pairs from original mnist
def get_data(
    key: PRNGKeyArray,
    n_rotations: int,
    n_pairs: int = N_PAIRS,
    collision_rate: float = 1.,
) -> Float[Array, 'pair (angle digit) (width height)']:
    angles = jnp.linspace(0, 360, n_rotations, endpoint=False)
    idxs_A, idxs_B = get_idxs(key, n_pairs, labels=labels)
    images_A, images_B = images[idxs_A], images[idxs_B]
    orbits_A = make_rotation_orbit(images_A, angles)
    orbits_A = ein.rearrange(orbits_A, 'pair angle w h -> (pair angle) w h')
    orbits_A = normalize_mnist(orbits_A)
    orbits_A = ein.rearrange(orbits_A, '(pair angle) w h -> pair angle w h',
        pair=n_pairs, angle=n_rotations)
    orbits_B = make_rotation_orbit(images_B, angles)
    orbits_B = ein.rearrange(orbits_B, 'pair angle w h -> (pair angle) w h')
    orbits_B = normalize_mnist(orbits_B)
    orbits_B = ein.rearrange(orbits_B, '(pair angle) w h -> pair angle w h',
        pair=n_pairs, angle=n_rotations)
    data, ps = ein.pack((orbits_A, orbits_B), 'pair * angle width height')
    return ein.rearrange(data, 'pair digit angle width height -> pair (angle digit) (width height)')


def get_deltasq(data: Float[Array, 'pair (angle digit) (width height)']) -> Float[Array, 'pair']:
    rearranged = ein.rearrange(data, 'pair (angle digit) wh -> digit pair angle wh', digit=2)
    projs = ein.reduce(rearranged, 'digit pair angle wh -> digit pair wh', 'mean')
    return ein.einsum((projs[0] - projs[1])**2, 'pair wh -> pair')


def get_projected_radius(data: Float[Array, 'pair (angle digit) (width height)']) -> Float[Array, 'pair']:
    rearranged = ein.rearrange(data, 'pair (angle digit) wh -> digit pair angle wh', digit=2)
    centers = ein.reduce(rearranged, 'digit pair angle wh -> digit pair wh', 'mean')
    radii_sq = ein.einsum((rearranged[..., 0, :] - centers)**2, 'digit pair wh -> digit pair')
    return ein.reduce(jnp.sqrt(radii_sq), 'digit pair -> pair', 'mean')


results_names = ( 'deltasq', 'sp_errs', 'em_errs', 'lambda_last', 'lambda_avg_no_last', 'lambda_avg', 'proj_radius', 'avg_angle', 'counts')
results_shape = (len(results_names), len(N_ROTATIONS), N_PAIRS)
results = np.empty(results_shape)
keys = jr.split(RNG, len(N_ROTATIONS))
for idx, (n_rot, key) in tqdm(enumerate(zip(N_ROTATIONS, keys)), total=len(N_ROTATIONS)):
    data = get_data(key, n_rot)
    # Empirical i.e. trained network error
    key, knet, ktrain = jr.split(key, num=3)
    knet = jr.split(knet, num=N_PAIRS)
    out_shape, params = jax.vmap(init_fn, in_axes=(0, None))(knet, data[0, 0].shape)
    # compute symmetrized empirical error
    x_train = data[:, 1:]
    y_train = ein.repeat( jnp.array([+1., -1.] * n_rot)[1:], 'angle -> pair angle', pair=N_PAIRS )
    params = kaiming_uniform_pytree(knet[0], params)  # works on ensembles too!
    params, losses = train( params, x_train, y_train, optim, epochs=N_EPOCHS, key=ktrain )
    pred_p = jax.vmap(apply_fn)(params, data[:, :1])
    em_err_p = jnp.abs(1-pred_p)

    x_train = data[:, :-1]
    y_train = ein.repeat( jnp.array([+1., -1.] * n_rot)[:-1], 'angle -> pair angle', pair=N_PAIRS )
    params = kaiming_uniform_pytree(knet[0], params)  # works on ensembles too!
    params, losses = train( params, x_train, y_train, optim, epochs=N_EPOCHS, key=ktrain )
    pred_m = jax.vmap(apply_fn)(params, data[:, -1:])
    em_err_m = jnp.abs(-1-pred_m)
    # empirical error is the mean between these two
    empirical_errors = ((em_err_p + em_err_m)/2).squeeze()
    empirical_correct_preds = ((pred_p > 0) + (pred_m < 0)).squeeze()

    # collect the rest of the stuff
    deltasq = get_deltasq(data)
    proj_radius = get_projected_radius(data)
    kernels = jax.vmap(kernel_fn)(data, data).ntk
    ckernels = jax.vmap(make_circulant)(kernels)
    spectral_errors = jax.vmap(circulant_error)(ckernels)
    # avg angle? (AR+BR)/2
    avg_angle = ein.reduce(ckernels[:, 0, 2::2], 'n k -> n', 'mean')
    # computation of elements of the inverse spectrum. NOTE THE REGULARIZATION
    # CONSTANT: see https://numpy.org/doc/stable/reference/routines.fft.html#normalization
    # isp = 1/jnp.abs(jax.vmap(orthofft)(ckernels[:, 0]) + REG*jnp.sqrt(2*n_rot))
    isp = 1/jnp.abs(jax.vmap(jnp.fft.fft)(ckernels[:, 0]))
    lambda_last = isp[:, n_rot]
    lambda_avg_no_last = ein.reduce(jnp.delete(isp, n_rot, axis=1), 'n d -> n', 'mean')
    lambda_avg = ein.reduce(isp, 'n d -> n', 'mean')
    # loading of results
    results[:, idx] = deltasq, spectral_errors, empirical_errors, lambda_last, lambda_avg_no_last, lambda_avg, proj_radius, avg_angle, empirical_correct_preds

np.save(res_path / 'results', results)
