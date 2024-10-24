# %%
import jax
from jax import numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import einops as ein

def kreg(
    k11: Float[Array, 'train train'],
    k12: Float[Array, 'train test'],
    k22: Float[Array, 'test test'],
    y: Float[Array, 'train 1'],
    reg: float = 1e-5
) -> tuple[Float[Array, 'test 1'], Float[Array, 'test 1']]:
    """
    Perform Gaussian Process regression.

    Args:
        k11: Kernel matrix between training points.
        k12: Kernel matrix between training and test points.
        k22: Kernel matrix between test points.
        y: Training labels.
        reg: Regularization parameter.

    Returns:
        mean: Predicted mean for test points.
        var: Predicted variance for test points.
    """
    sol = jax.scipy.linalg.solve(k11 + reg*jnp.eye(len(k11)), k12, assume_a='pos')
    mean = ein.einsum(sol, y, 'train test, train d-> test d')
    var = k22 - ein.einsum(sol, k12, 'train t1, train t2 -> t1 t2')
    return mean, var


def circulant_error(k: Float[Array, 'n n'], reg: float = 1e-5) -> float:
    """
    Calculate the prediction error for a 2-class setting with interleaved points.

    This function assumes a scenario where there are two classes, each with N elements,
    and the points are interleaved (i.e., the classes alternate in the sequence).
    The error is calculated based on the circulant structure of the kernel matrix.

    Args:
        k: The kernel matrix.
        reg: Regularization parameter.

    Returns:
        float: The predicted error for the interleaved 2-class setting.
    """
    isp = 1 / jnp.abs(jnp.fft.fft(k[0]) + reg)
    return float(isp[len(isp)//2]/jnp.mean(isp))


def make_circulant(k: Float[Array, 'n n']) -> Float[Array, 'n n']:
    """
    Convert a covariance matrix to a circulant matrix by averaging the diagonals.

    Args:
    k (jnp.ndarray): Input covariance matrix.

    Returns:
    jnp.ndarray: Circulant matrix derived from the input covariance matrix.
    """
    n = len(k)
    aligned = jax.vmap(jnp.roll)(k, -jnp.arange(n))
    means = jnp.mean(aligned, axis=0)
    out = jax.vmap(jnp.roll, in_axes=(None, 0))(means, jnp.arange(n))
    return out


# %% Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set up test parameters
    N = 11  # keep this to even values
    L = 10
    reg = 1e-5

    # Generate test data
    xs = jnp.linspace(-1, 1, N)[:, None]
    ys = jnp.array([-1., 1.] * (N//2) + [-1]*(N%2))[:, None]

    # Create kernel matrix
    k = jnp.exp(-L**2 * jnp.sum((xs - xs[:, None])**2, axis=-1))
    k = make_circulant(k)

    # Prepare train-test split
    k_train_train = jnp.delete(jnp.delete(k, N//2, axis=0), N//2, axis=1)
    k_train_test = jnp.delete(k[:, N//2:N//2+1], N//2, axis=0)
    k_test_test = k[N//2:N//2+1, N//2:N//2+1]
    y_train = jnp.delete(ys, N//2, axis=0)

    # Perform GP regression
    y_pred, y_var = kreg(k_train_train, k_train_test, k_test_test, y_train, reg)
    error = jnp.abs(ys[N//2]-y_pred)
    print(f"Prediction error at missing point: {error[0, 0]:.2f} pm {jnp.sqrt(y_var)[0, 0]:.2f}")

    # Check error
    error_from_formula = circulant_error(k, reg=reg)
    print(f'Prediction from formula: {error_from_formula:.2f}')

    # Generate predictions on a finer grid for plotting
    xt = jnp.delete(xs, N//2, axis=0)
    yt = jnp.delete(ys, N//2, axis=0)
    xf = jnp.linspace(-1, 1, 100)[:, None]
    k_train_fine = jnp.exp(-L**2 * jnp.sum((xf - xt[:, None])**2, axis=-1))
    k_fine_fine = jnp.exp(-L**2 * jnp.sum((xf - xf[:, None])**2, axis=-1))
    yf_mean, yf_var = kreg(k_train_train, k_train_fine, k_fine_fine, y_train, reg)
    yf_std = jnp.sqrt(jnp.diag(yf_var))
    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(xf, yf_mean, label='GP Mean')
    plt.fill_between(xf.flatten(),
                     yf_mean.flatten() - 2*yf_std,
                     yf_mean.flatten() + 2*yf_std,
                     alpha=0.3, label='95% Confidence')
    plt.scatter(xt, yt, c='red', label='Training Data')
    plt.scatter(xs[N//2], ys[N//2], c='green', s=100, label='Test Point')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
