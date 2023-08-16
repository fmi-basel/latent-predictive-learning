#!/bin/sh

for FREQ in 0.1 1 5 10 20 30 40 50; do 
	python mk_spikes.py --freq $FREQ
	./run_stdp.sh 
	cp ~/data/lpl/stdp/input_con.0.wmat ~/data/lpl/stdp/input_con_${FREQ}hz.0.wmat
done


# different initial conditions
for FREQ in 0.1 1 5 10 20 30 40 50; do 
	python mk_spikes.py --freq $FREQ
	for MEAN in 0 20 50; do
		for SIGMA2 in 0.1; do
			./run_stdp.sh $MEAN $SIGMA2 $FREQ
		done
	done
done

