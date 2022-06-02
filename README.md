# Latent Predictive Learning

![Alt text](summary_figure.png "LPL")

This repository contains code for reproducing results reported in our preprint on [LPL](https://www.biorxiv.org/content/10.1101/2022.03.17.484712v1), a framework for biologically plausible self-supervised learning.

## Setup

The deep learning simulations are based on the pytorch lightning framework. We've provided a requirements file to help you install everything you need by just running:
```
pip install -r requirements.txt
```

We recommend that you install these into a separate project-specific virtual environment.

For the spiking simulations, you'll need to install [Auryn](https://www.fzenke.net/auryn/doku.php).

## Usage

To train a deep net with layer-local LPL, simply run

```
python lpl_main.py
```

in the virtual environment you just created. Several useful command-line arguments are provided in `lpl_main.py` and `models\modules.py`. A few are listed below:
- `--train_with_supervision` trains the same network with supervision.
- `--use_negative_samples` trains the network with a cosine-distance-based contrastive loss. 
**Note:** this needs to be combined with setting the decorrelation loss coefficient to 0, and enabling the additional projection MLP.
- `--train_end_to_end` is a flag for training the network with backpropagation to optimize the specified loss (lpl, supervised or neg. samples) at the output layer only.
- `--no_pooling` optimizes the specified loss on the unpooled feature maps.
- `--use_projector_mlp` adds additional projection dense layers at every layer in the network where the specified loss is to be optimized.
- `--pull_coeff`, `--push_coeff`, and `--decorr_coeff` are the different loss coefficients. Default values are 1.0, 1.0, and 10.0 respectively.

### Note on network architectures
Only a VGG-11 architecture is provided here, but the framework can easily be extended to other architectures. You can simply configure another encoder in `models\encoders.py`, and add it to `models\network.py`. In principle, everything should work with residual architectures as well, but layer-local learning in this case is not well-defined. 

## Analysis

Jupyter notebooks (3a, 3b and 3c) provided in `notebooks` contain instructions on extracting and visualising several metrics for the quality of learned representations. Also provided there are notebooks for generating all figures from the paper.

## Citation

```
@article{halvagal2022combination,
  title={The combination of Hebbian and predictive plasticity learns invariant object representations in deep sensory networks},
  author={Halvagal, Manu Srinath and Zenke, Friedemann},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```