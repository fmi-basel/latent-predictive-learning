#!/bin/sh

# Use longer max pre-post interval of 50ms 

# different initial conditions
FREQ=10
python mk_spikes.py --interval 50e-3 --freq $FREQ
for MEAN in 0 5 10 12 15 20 30; do
	for SIGMA2 in 0.0 0.001 0.005 0.01 0.1 0.2 1.0 100; do
		./run_stdp_long.sh $MEAN $SIGMA2 $FREQ &
	done
	wait
done

