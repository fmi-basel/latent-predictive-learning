Here we keep track of typos and errors that have been reported to us by the attentive readers of our article:

> Halvagal, Manu Srinath, and Friedemann Zenke. 2023. ‘The Combination of Hebbian and Predictive Plasticity Learns Invariant Object Representations in Deep Sensory Networks’. Nature Neuroscience, October, 1–10. [https://doi.org/10.1038/s41593-023-01460-y](https://doi.org/10.1038/s41593-023-01460-y).

Thanks for reporting these problems. Our sincerest apologies that they made it into the final manuscript.


# Spiking Network Simulations

Thanks to Github user [yilun-wu](https://github.com/yilun-wu), who made us aware of two small discrepancies between our simulation code and its description in the methods (Issues [#2](https://github.com/fmi-basel/latent-predictive-learning/issues/2), [#3](https://github.com/fmi-basel/latent-predictive-learning/issues/3)), and ([Issue #5](https://github.com/fmi-basel/latent-predictive-learning/issues/5))
.


## Implementation of transmitter triggered plasticity ([Issue #2](https://github.com/fmi-basel/latent-predictive-learning/issues/2))

First, there is a small mismatch between our spiking learning rule simulation and the rule reported in the Methods Eq. (18).
The learning rule we simulated was the following and that's what Eq. (18) should have read:
$$\frac{\mathrm{d}w_{ij}}{\mathrm{d}t} = \eta~\alpha \ast \left( \epsilon \ast S_j(t) f^\prime(U_i(t)) \right) \left[ \alpha \ast \left( -\left( S_i(t) - S_i(t-\Delta t))\right) + \frac{\lambda}{\sigma_i^2 + \xi} \left( S_i(t) - \bar S_i(t) \right) + \delta  \right) \right]$$
whereas we wrote:
$$\frac{\mathrm{d}w_{ij}}{\mathrm{d}t} = \eta~\alpha \ast \left( \epsilon \ast S_j(t) f^\prime(U_i(t)) \right) \left[ \alpha \ast \left( -\left( S_i(t) - S_i(t-\Delta t))\right) + \frac{\lambda}{\sigma_i^2 + \xi} \left( S_i(t) - \bar S_i(t) \right) \right) \right] + \eta \delta S_j(t)$$

Importantly, the plasticity rule we simulated corresponds to an alternative implementation of transmitter-triggered plasticity, which **does not** affect the results, which remain qualitatively unchanged. 
This qualitative resemblance can be seen in the following animated GIF comparing Figure 5 as published and three versions in which we [patched](patches/alternative_transmitter_triggered.patch) the code and simulated Eq. (18) as reported in the methods for varying values of $\delta$.

![Figure 5 comparison for original and patched code.](figs/altern_trans_trig_plast.gif "Fig5comp")


## Implementation of double exponential filtering for synaptic traces ([Issue #3](https://github.com/fmi-basel/latent-predictive-learning/issues/3))

There is another mismatch between code as simulated and the methods which affects the effective learning rate of the spiking rule.
Specifically, the double exponential filtering with $\epsilon$ as well as with the $\alpha$ kernel on the left in Eq. (18) above was implemented as follows:
$$\frac{\mathrm{d}\bar c}{dt}(t) = -\frac{\bar c(t)}{\tau^\mathrm{rise}} + c(t)$$

$$ \tau^\mathrm{fall} \frac{\mathrm{d} \bar{\bar{c}}}{\mathrm{d}t}(t) = -\bar{\bar{c}}(t) + \bar c(t) $$

in contrast to what we stated in the methods, which was

$$\tau^\mathrm{rise} \frac{\mathrm{d}\bar c}{dt}(t) = -\bar c(t) + c(t)$$

$$\tau^\mathrm{fall} \frac{\mathrm{d} \bar{\bar{c}}}{\mathrm{d}t}(t) = -\bar{\bar{c}}(t) + \bar c(t)$$

The rhs convolution with $\alpha$ was implemented as stated.
Since all filters are implemented through linear ODEs, the difference in implementation corresponds to an amplitude change of the filtered quantity while the shape remains unaffected. 
Thus the change corresponds to a change of learning rate by a factor of $\tau_\mathrm{rise}$. Hence the effective learning rate was by this factor _lower_ in our simulations than the methods suggest. 
Apologies for these inconsistencies. 


## Implementation of relative refractory period ([Issue #5](https://github.com/fmi-basel/latent-predictive-learning/issues/5))

In the implementation of the absolute and relative refractory period, the moving threshold of our neuron model is set to 50mV after every spike before being exponentially decayed down to the resting threshold value of -50mV instead of jumping by 100mV as stated in the paper (cf. Eq. (16)). Thus threshold effects do not accumulate, but are reset with every spike. However, this difference only causes minor differences at *very* high firing rates to the model as described in the methods.

