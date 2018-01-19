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

The [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test) test can then be computed.  First, generate an object from which the cumulative distribution function (CDF) for the mixture can be calculated.  We provide an object like a `scipy.stats` distribution that has methods for the CDF, as well as probability distribution function (PDF) and random variable (RV) generation:
```python
import scipy.stats
np_dist=nulling_mcmc.MultiGaussian(means_fit,stds_fit,weights_fit)
print scipy.stats.kstest(on,np_dist.cdf)
```
which will print the K-S test statistic as well as the associated p-value.  The [Anderson-Darling](https://en.wikipedia.org/wiki/Anderson–Darling_test) is more sensitive, but has the disadvantage the p-values have to be computed empirically for each mixture.  This can also be done.  First we simulate 1000 data-sets drawn from our best-fit distribution and compute the test statistic for each
```
Nsim=1000
A2sim=np.zeros(Nsim)
ysim=np_dist.rvs((Nsim,len(on)))
for j in xrange((Nsim)):
    A2sim[j]=nulling_mcmc.anderson_darling(ysim[j], np_dist.cdf)
```
Then we compare against the value for the actual data and compute a p-value:
```python
print (A2sim>nulling_mcmc.anderson_darling(on, np_dist.cdf)).sum()/float(Nsim)
```

