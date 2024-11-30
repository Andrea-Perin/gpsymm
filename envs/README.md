# Environments

There are three environments:
* `env.yml`: mostly `JAX` stuff, which the majority of the code requires. This one *needs* GPU.
* `macenv.yml`: basically the same as above, but for machines without GPU support.
* `torchenv.yml`: contains the pytorch stuff that you'll need if you want to reproduce the results in Figure 1.
