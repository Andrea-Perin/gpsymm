import gzip
from jax import numpy as jnp
from jaxtyping import Float, Array, UInt
import einops as ein


def load_images(img_path):
    with gzip.open(img_path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = jnp.frombuffer(image_data, dtype=jnp.uint8)\
            .reshape((image_count, row_count, column_count))
        return images / 255.


def load_labels(lab_path):
    with gzip.open(lab_path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = jnp.frombuffer(label_data, dtype=jnp.uint8)
        return labels


def normalize_mnist(mnist: UInt[Array, '60000 28 28']) -> Float[Array, '60000 28 28']:
    """
    Normalize MNIST dataset images.

    This function performs the following normalization steps:
    1. Scales pixel values to the range [0, 1].
    2. Subtracts the mean of each image.
    3. Projects each image onto a unit sphere.

    Args:
        mnist (UInt[Array, '60000 28 28']):
            The MNIST dataset as an unsigned integer array.
            Shape: (60000, 28, 28), where:
                - 60000 is the number of images
                - 28x28 is the size of each image

    Returns:
        Float[Array, '60000 28 28']:
            The normalized MNIST dataset as a float array.
            Each image is centered around zero and projected onto a unit sphere.

    Note:
        This normalization ensures that all images have the same L2 norm,
        which can be beneficial for certain machine learning algorithms.
    """
    mnist /= 255.
    mean = ein.reduce(mnist, 'n w h -> n 1 1', 'mean')
    mnist = mnist - mean
    # project on sphere
    norm = jnp.sqrt(ein.reduce(mnist**2, 'n w h -> n 1 1', 'sum'))
    return mnist/norm


# %%
if __name__ == "__main__":
    from pathlib import Path
    labpath = Path('./data/MNIST/raw/train-labels-idx1-ubyte.gz')
    imgpath = Path('./data/MNIST/raw/train-images-idx3-ubyte.gz')
    images = load_images(img_path=imgpath)
    normimgs = normalize_mnist(images)
    check = ein.einsum(normimgs**2, 'n w h -> n')
    assert jnp.allclose(jnp.ones_like(check), check)
