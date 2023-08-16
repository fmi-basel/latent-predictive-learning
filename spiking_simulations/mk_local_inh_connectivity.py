#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *
from scipy.io import mmwrite


nb_pre = 100
nb_post = nb_pre//4
sigma = 20.0 # Width of diagonal in channels 
 
mat = np.empty((nb_pre,nb_post))

x = np.arange(nb_pre)
for i,c in enumerate(np.linspace(0,nb_pre,nb_post)):
    mat[:,i] = np.exp(-(x-c)**2/sigma**2)

# binarize mask
rnd = np.random.rand(*mat.shape)
con = 1.0*(mat>rnd)

# save to file
sw = csr_matrix(con) # ensures row major format in COO output
mmwrite('local_inh_connectivity.wmat',sw)


# Plot connectivity for visual inspection
plt.imshow(con)
plt.xlabel("post")
plt.ylabel("pre")
plt.savefig("local_inh_connectivity.pdf")

