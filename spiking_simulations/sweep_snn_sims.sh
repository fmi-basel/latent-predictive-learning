#!/bin/sh


DIR=`pwd`
for SEED in 123; do 
	for TAU in 600; do 
		for LAMBDA in 0 1; do 
			for PHI in 0 1; do
				echo "cd $DIR; ./run_snn_augmented_grad.sh $LAMBDA $PHI $TAU $SEED"
				echo "cd $DIR; ./run_snn.sh $LAMBDA $PHI $TAU $SEED"
				echo "cd $DIR; ./run_snn_fixed_inh.sh $LAMBDA $PHI $TAU $SEED"
				echo "cd $DIR; ./run_snn_feedforward.sh $LAMBDA $PHI $TAU $SEED"
			done
		done
	done
done

