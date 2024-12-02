# GPSYMM

Code for the paper.
Here a list of what each file/directory is supposed to do:
* `envs`: contains the environments that are needed;
* `utils`: contains various utilities (from Gaussian Process solving to our formulas);
* `images`: contains the images produced by the various files;
* `results`: contains the numerical results for "heavier" files, for which the plotting is on separate files;
* `data`: contains the raw data that is used all around the project. It is generated when running `fig1_data.py`;
* `config.toml`: parameters for the various experiments;
* `fig1_data.py`: produces the data that is used in Figure 1. This file is "special" because it requires the `torchenv` enviroment;
* `fig1_plot.py`: using the data from `fig1_plot.py`, produces the corresponding plots;
* `toy_model.py`: produces the plot with the dependency of the error for a simple kernel on delta and N;
* `spectrum_sketch.py`: produces the plot which shows a kernel and its spectrum, together with the kernel/distance matrices;
* `mlp_ntk.py`: produces the data for a comparison of spectral and NTK empirical errors for an MLP. The MLP can be made deeper with a flag (run `mlp_ntk.py -h` for more info);
* `cntk.py`: produces the data for a comparison of spectral and NTK empirical errors for a CNN. The CNN can be made fully connected (fc) or with a Global Average Pooling (GAP) with a flag (run `cntk.py -h` for more info);
* `trained_mlp.py`: produces the data for a comparison of spectral and trained empirical errors for a 1-hidden layer MLP;
* `plot_ntk_analysis.py`: takes the data produced by `mlp_ntk.py`, `cntk.py`, `trained_mlp.py` and turns it into figures;
* `multiple_seeds.py`: produces the plot that compares the NTK empirical and spectral errors for the multiple seed case for various angles;
* `m_classification.py`: produces the plot that compares the spectral and the trained errors for the multiple class case.
* `myplots.mlpstyle`: `matplotlib` style sheet for the plots.

## Env guide
There are three envs.
For `fig1_data.py` only, use `torchenv`.
For the rest, use `gpsymmenv` (if you have a GPU) or `macenv` if you don't.
