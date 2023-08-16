#!/usr/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot weight histogram.')
parser.add_argument('--freq', type=float, default=1.0, help='repetitation freq')
parser.add_argument('--pairings', type=int, default=100, help='nb pairings')
parser.add_argument('--interval', type=float, default=20e-3, help='maximum pairing pre-post interval')
args = parser.parse_args()

timestep = 1e-4
nb_pairings = args.pairings

pre_size = 100
interval = args.interval
dts = np.linspace(-interval,interval,pre_size)
offset = 100e-3

spkfile = open("spikes.ras",'w')
curfile = open("post_voltage.dat",'w')

U_spk = -20e-3
U_dep = -51e-3
U_hyp = U_dep
U_rest = -60e-3

dep_interval = 10e-3
ref_interval = 5e-3

spike_times = []
spike_ids = []

curfile.write("%e %e\n"%(0.0, U_hyp))
for p in range(nb_pairings):
    spike_time = offset+p/args.freq
    for j,dt in enumerate(dts):
        spike_times.append(spike_time+dt)
        spike_ids.append(j)

    curfile.write("%e %e\n"%(spike_time-dep_interval-timestep, U_hyp))
    curfile.write("%e %e\n"%(spike_time-dep_interval, U_dep))
    curfile.write("%e %e\n"%(spike_time-2*timestep, U_dep))
    curfile.write("%e %e\n"%(spike_time-timestep, U_spk))
    curfile.write("%e %e\n"%(spike_time, U_spk))
    curfile.write("%e %e\n"%(spike_time+timestep, U_dep))
    curfile.write("%e %e\n"%(spike_time+ref_interval, U_hyp))

curfile.write("%e %e\n"%(spike_time+2*ref_interval, U_rest))
curfile.write("%e %e\n"%(spike_time+200.0, U_rest))


# Sort spikes in case of overlap
idx = np.argsort(np.array(spike_times))
spike_times = np.array(spike_times)[idx]
spike_ids = np.array(spike_ids)[idx]

# Write spikes to file
for t,i in zip(spike_times, spike_ids):
    spkfile.write("%e %i\n"%(t, i))
