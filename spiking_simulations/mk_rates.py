#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dt = 10e-3 # sampling interval
baserate = 5.0
amplitude = 4.0
basename = "rates"

np.random.seed(42)

def get_random_signal(period=10.0, alpha=1.5, nb_repeats=11, cutoff=128):
    theta = np.random.rand(cutoff,2)
    duration = period*nb_repeats
    times = np.arange(0,duration,dt)
    signal = 0.0
    for i,params in enumerate(theta):
        a,b = params
        signal += 1.0/alpha**i*a*np.sin(2*np.pi*i*(times+b)/period)

    signal = (signal-np.mean(signal))/(np.std(signal)+1e-4)
    return times,signal


#Make rates0.dat with constant firing rate
duration = 100.0
with open("rates0.dat",'w') as fp:
    fp.write("0 %e\n"%baserate)
    fp.write("%e %e\n"%(duration,baserate))

plt.figure()

filecount = 1
# Make periodic slow signals and controls
alphas = [1.1, 1.1]
multipliers = [1.0, 1.0]
for k in range(len(alphas)):
    period = 3.0+k*1/13
    times,signal = get_random_signal(period=period,alpha=alphas[k])
    values = np.clip(multipliers[k]*amplitude*signal+baserate,0.1,np.inf)
    with open("%s%i.dat"%(basename,filecount),'w') as fp:
        for p in zip(times,values):
            fp.write("%e %e\n"%p)
    plt.plot(times,values)
    filecount += 1

    # Make shuffle control
    np.random.shuffle(values)
    with open("%s%i.dat"%(basename,filecount),'w') as fp:
        for p in zip(times,values):
            fp.write("%e %e\n"%p)
    plt.plot(times,values,label=r'$\alpha=%f,\sigma=%f$'%(alphas[k],multipliers[k]))
    filecount += 1


# Plot connectivity for visual inspection
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.xlim((0,3.0))
plt.legend(fontsize=8)
plt.tight_layout()
sns.despine()
plt.savefig("rates.pdf")

