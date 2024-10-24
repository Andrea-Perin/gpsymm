# %%
from jax import numpy as jnp, random as jr, vmap
from jaxtyping import Array, Float
from typing import Callable


def xshift_img(img: Float[Array, 'w h'], sfrac: float) -> Float[Array, 'w h']:
    """
    Shift an image horizontally in the Fourier domain.

    This function performs a horizontal shift of an image by modifying its Fourier transform.
    The shift is performed by applying a phase shift in the frequency domain, which allows
    for sub-pixel shifts without interpolation artifacts.

    Args:
        img: A 2D array representing the input image with shape (width, height).
        sfrac: The fraction of image width to shift by. Positive values shift right,
               negative values shift left. A value of 1.0 shifts by the full image width.

    Returns:
        The shifted image as a 2D array of the same shape as the input.

    Note:
        This function uses the FFT method for shifting, which assumes periodic boundary
        conditions. Content that shifts past one edge will appear on the opposite edge.
    """
    w = img.shape[1]
    fftimg = jnp.fft.rfft2(img)
    mag, ph = jnp.abs(fftimg), jnp.angle(fftimg)
    phases = (2*jnp.pi*sfrac) * jnp.arange(w//2+1)
    return jnp.fft.irfft2(mag*jnp.exp(-1j*(ph+phases)))


def kronmap(fn: Callable, nargs: int) -> Callable:
    """
    Create a Kronecker-mapped version of a function.

    This decorator applies multiple vmaps to a function to create a Kronecker product-like
    mapping over its arguments. The resulting function broadcasts the original function
    over all possible combinations of its inputs.

    Args:
        fn (Callable): The function to be Kronecker-mapped.
        nargs (int): The number of arguments in the function to be mapped over.

    Returns:
        Callable: A new function that applies the original function in a Kronecker product-like manner.

    Example:
        If fn(a, b, c) is a function of three arguments, then:
        kronmapped = kronmap(fn, 3)
        result = kronmapped(a, b, c)

        This is equivalent to:
        result[i, j, k] = fn(a[i], b[j], c[k])

        for all possible combinations of i, j, and k.

    Note:
        - The decorated function will have the same number of arguments as the original function.
        - Each argument should be an array-like object that can be indexed.
        - The output will have a shape that is the concatenation of all input shapes.

    Implementation details:
        The function successively applies vmap to each argument, creating a chain of
        mapped functions. Each application of vmap introduces a new batch dimension
        in the output, corresponding to one of the input arguments.
    """
    for i in range(nargs):
        fn = vmap(
            fn,
            in_axes=(None,) * i + (0,) + (None,) * (nargs-i-1),
            out_axes=i
        )
    return fn


def three_shear_rotate(img: Float[Array, 'd d'], theta: float) -> Float[Array, 'd d']:
    """
    Perform image rotation using the three-shear method.
    This function performs a quick and interpolation-free rotation of a square,
    single-channel image using the three-shear method. It conserves the pixels
    by shifting them across the image without interpolating them. The function
    supports batch processing via `vmap` for efficient computation on multiple
    images.
    Parameters
    ----------
    img : Float[Array, 'd d']
        A 2D array representing the input image, assumed to be square-shaped
        and single-channel. The image is rotated in-place, where `d` is the
        dimension of the image (width and height).

    theta : float
        The angle of rotation in radians.
    Returns
    -------
    Float[Array, 'd d']
        The rotated image as a 2D array of the same shape as the input `img`.
        Pixels outside the boundary of the original image are set to 0.
    Notes
    -----
    This method uses three shear transformations to rotate the image without
    interpolation. The process ensures that no pixel values are altered, only
    repositioned.
    Assumptions
    -----------
    - The input image is square-shaped.
    - The image has a single channel (grayscale).

    This method may not work as intended for images that do not meet these
    assumptions.
    TODO
    ----
    - Rewrite this function with `einops` instead of the current reshaping
      strategy to improve clarity and reduce reliance on manual reshaping.
    References
    ----------
    - See Tom Forsyth's article on the three-shear rotation method:
      https://cohost.org/tomforsyth/post/891823-rotation-with-three

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Float, Array
    >>> img = jnp.ones((100, 100))
    >>> rotated_img = three_shear_rotate(img, jnp.pi / 4)
    >>> rotated_img.shape
    (100, 100)
    """
    w, h = img.shape
    s = w // 2  # assumes square images
    shear_tan = jnp.array([[1, -jnp.tan(theta/2)], [0, 1]])
    shear_sin = jnp.array([[1, 0], [jnp.sin(theta), 1]])
    og_grid = jnp.stack(jnp.meshgrid(
        jnp.arange(-s, s),
        jnp.arange(-s, s))).T
    new_grid = shear_tan @ shear_sin @ shear_tan @ (og_grid.reshape(-1, 2).T)
    qng = new_grid.T.reshape(h, w, 2).astype(int)
    rotated = img[qng[..., 0]+s, qng[..., 1]+s]
    return jnp.where((jnp.abs(qng) > s).any(-1), 0, rotated)


def sgd_dataloader(xs, ys, batch_size, *, key):
    """
    Generates batches of data for Stochastic Gradient Descent (SGD).

    Args:
        xs (jax.numpy.ndarray): Input features.
        ys (jax.numpy.ndarray): Corresponding labels.
        batch_size (int): Size of each batch.
        key (jax.random.PRNGKey): Key for random number generation.

    Yields:
        tuple: A tuple containing the batch of input features and corresponding labels.

    Raises:
        ValueError: If the lengths of input features and labels do not match.

    Notes:
        This function generates batches of data for stochastic gradient descent (SGD).
        It shuffles the dataset at each iteration and yields batches of specified size.

    Example:
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>> key = random.PRNGKey(0)
        >>> xs = jnp.array([[1, 2], [3, 4], [5, 6]])
        >>> ys = jnp.array([0, 1, 0])
        >>> dataloader = sgd_dataloader(xs, ys, batch_size=2, key=key)
        >>> for batch_xs, batch_ys in dataloader:
        ...     print(batch_xs, batch_ys)
        [[3 4]
         [1 2]] [1 0]
        [[5 6]] [0]
    """
    if len(xs) != len(ys):
        raise ValueError(f"Wrong data shapes: len(xs)={len(xs)}, len(ys)={len(ys)}")

    dataset_size = xs.shape[0]
    indices = jnp.arange(dataset_size)

    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)

        start = 0
        end = batch_size

        while end < dataset_size:
            batch_perm = perm[start:end]
            yield xs[batch_perm], ys[batch_perm]
            start = end
            end = start + batch_size


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def testfun(a, b):
        return a*b

    krontest = kronmap(testfun, 2)
    a = jnp.linspace(0, 1, 100)
    b = jnp.linspace(1, 2, 123)
    out = krontest(a, b)
    assert out.shape == (len(a), len(b))
    assert out[1, 2] == a[1]*b[2]

    # shift test
    img = jnp.eye(10)
    shifts = jnp.linspace(0, 1, 20, endpoint=False)
    shifted_imgs = vmap(xshift_img, in_axes=(None, 0))(img, shifts)
    # check integer pixel shifts
    plt.imshow(shifted_imgs[0])
    plt.show()
    plt.imshow(shifted_imgs[2])
    plt.show()
    plt.imshow(shifted_imgs[-2])
    plt.show()
    # check fractional shifts
    plt.imshow(shifted_imgs[1])
    plt.colorbar()
    plt.show()
