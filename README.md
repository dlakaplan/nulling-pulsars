# nulling-pulsars

Gaussian Mixture model for analyzing nulling pulsars.  Uses [emcee](http://dfm.io/emcee/current/) to do Markov Chain Monte Carlo fit to pulse intensities, where intensities measuring in an OFF-pulse phase bin are used to constrain the NULL.  There can be an arbitrary number of components, and they can have arbitrary distributions that need not be the same, although Gaussians are the default.

## Example:
Assume there are arrays `on` and `off` containing the pulse intensities in ON-pulse and OFF-pulse phase bins.  Then:

```python
NP=nulling_mcmc.NullingPulsar(on, off, 2)

means_fit, means_err, stds_fit, stds_err, weights_fit, weights_err, samples, lnprobs=NP.fit_mcmc(nwalkers=nwalkers,
                                                                                                 niter=niter,
                                                                                                 ninit=50,
                                                                                                 nthreads=nthreads,
                                                                                                 printinterval=50)
```
will compute the model for `niter` iterations (500 works well) and using `nwalkers` walkers (40 works well).  The results can be exampled using a corner plot:
```python
import corner
corner.corner(samples)
```

The model parameters can be used to calculate the probabilities that individual pulses are due to the NULL component:
```python
null_prob=NP.null_probabilities(means_fit, stds_fit, weights_fit, NP.on)
```

The Akaike and Bayesian information criteria (AIC and BIC) can also be calculated to compare the results with varying numbers of components.  For instance 1 component would be for a pulsar that does not null, while multi-mode pulsars could have 3 or more components.
```python
AIC=NP.AIC(means_fit,stds_fit,weights_fit)
BIC=NP.BIC(means_fit,stds_fit,weights_fit)
```
