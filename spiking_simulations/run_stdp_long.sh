#!/bin/bash


TMPDIR=`mktemp -d`
mkdir -p $TMPDIR
OUTDIR=~/data/lpl/stdp
mkdir -p $OUTDIR
echo "Tempdir $TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
# ln -s $TMPDIR/ out/

make 

SIMTIME=200
LAMBDA=1

if [ "$1" != "" ]; then
	MEAN=$1
else
	MEAN=20
fi

if [ "$2" != "" ]; then
	SIGMA2=$2
else
	SIGMA2=0.1
fi

if [ "$3" != "" ]; then
	FREQ=$3
else
	FREQ=20
fi


echo "./mk_spikes.py --interval 50e-3 --freq $FREQ"
./mk_spikes.py --interval 50e-3 --freq $FREQ
nice ./sim_lpl_stdp_protocol --prime 0.0 --simtime $SIMTIME --dir $TMPDIR --sparsenes 1.0 --wext 0.5 --wee 0.0 --psize 100 --plen 4 --tau_mean 600 --tau_sigma2 20 --lambda $LAMBDA --eta 5e-3 --initmean $MEAN --initsigma2 $SIGMA2
cp $TMPDIR/* $OUTDIR
for EXT in wmat syn; do
	cp $TMPDIR/input_con.0.$EXT $OUTDIR/input_con_mean-${MEAN}_sigma2-${SIGMA2}_fr-${FREQ}_long.$EXT
done
# python plot_weight_hist.py --input $TMPDIR/input_con.0.wmat --output $OUTDIR/weights_lam${LAMBDA}.pdf

