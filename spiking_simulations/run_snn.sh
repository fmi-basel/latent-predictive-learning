#!/bin/bash


if [ "$1" != "" ]; then
	LAMBDA=$1
else
	LAMBDA=1
fi

if [ "$2" != "" ]; then
	PHI=$2
else
	PHI=1
fi

if [ "$3" != "" ]; then
	TAU=$3
else
	TAU=600
fi

if [ "$4" != "" ]; then
	SEED=$4
else
	SEED=123
fi

make 

EXPNAME='spiking'
SIMTIME=100000
ETA=1e-2
ZETA=1e-3
KAPPA=10
WEE=0.0
WEI=0.5
WIE=0.5
WII=0.4
TAUW=1e9
TAUS2=30
AMPA=1
FLAGS=" --ieplastic "

OUTPUTDATADIR="$HOME/data/lpl/${EXPNAME}/tau${TAU}_lam${LAMBDA}_phi$PHI/recurrent/"
mkdir -p $OUTPUTDATADIR


export HWLOC_COMPONENTS=-gl

nice ./sim_lpl_spiking --quiet --seed $SEED --simtime $SIMTIME --dir $OUTPUTDATADIR --sparseness 0.1 --ampa $AMPA --ne 100 --ni 25 --wext 0.15 --wee $WEE --wei $WEI --wii $WII --wie $WIE --psize 100 --plen 4 --tau_mean $TAU --tau_sigma2 $TAUS2 --phi $PHI --lambda $LAMBDA --zeta $ZETA --kappa $KAPPA --eta $ETA --tau_weight_decay $TAUW $FLAGS
