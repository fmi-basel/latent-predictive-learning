# Code to reproduce Spiking LPL results

This directory contains the code to reproduce all spiking network results with Latent Predictive Learning.

## Compiling the code

The spiking network simulations require the [Auryn Simulator](https://fzenke.net/auryn). 
You can download and compile the required version of Auryn by running the following commands in your home directory:

```
git clone https://github.com/fzenke/auryn.git
cd auryn
git checkout -b dev 36b3c197
cd build/release
./bootstrap.sh && make
```

In the following we assume you have downloaded and compiled Auryn under
`~/auryn`, otherwise please adjust `AURYNDIR` in `Makefile` to your local
installation. 


### Install required Python packages

A python environment with numpy, scipy and matplotlib is also required for the analysis.
The python environment created for running the deep network simulations would be sufficient, so make sure to activate it first for all the following steps, and for the analysis in the `notebooks` directory.
We also provide a minimal `requirements.txt` file here to install only the necessary packages with pip.

`pip install -r requirements.txt`


### Compile required simulation files

To compile the code run `make` in this directory. 

## Run Network simulations 

The network simulation is implemented in `sim_lpl_spiking.cpp` with the corresponding run files:

* `run_snn.sh`: This runs the standard network simulation with recurrent plastic inhibition.
* `run_snn_augmented_grad.sh`: This runs the same network but using an
  optimizer, which allows to terminate the simulation earlier. We did not use
  this in the paper, but it's useful to speed things up when playing around.
* `run_snn_feedforward.sh`: The network without inhibition.
* `run_snn_fixed_inh.sh`: The network with fixed recurrent inhibition.

For reading the data and generating figures see `notebooks/5 - LPL spiking.ipynb`.


## Run STDP induction protocol simulations

The STDP protocol is implemented in `sim_lpl_stdp_protocol.cpp` with the corresponding run files:

* `mk_spikes.py`: Creates necessary files for STDP protocols (run this script
  first, but it also called by the sweep scripts directly).
* `sweep_stdp_long.sh`: Creates data for STDP curve plots.
* `sweep_stdp.sh`: Creates data for the rate dependence of STDP plot.

For reading the data and generating figures see `notebooks/6 - STDP.ipynb`.
