from __future__ import print_function, division

import numpy as np, numpy.random
import emcee
import scipy.stats


def gaussian(x, params):
    """
    y=gaussian(x, params)
    params=[mean, std]
    return a Gaussian distribution with mean and standard deviation evaluated at x
    """
    mean, std = params
    return np.exp(-((x - mean) ** 2) / 2.0 / std ** 2) / np.sqrt(2 * np.pi * std ** 2)


def anderson_darling(y, CDF):
    """
    anderson_darling(y, CDF)

    returns the anderson darling test statistic A^2
    for the data y using callable CDF
    https://en.wikipedia.org/wiki/Anderson–Darling_test
    """

    y = np.sort(y)
    n = len(y)
    i = np.arange(len(y))
    S = (
        (((2 * i + 1) / float(n)) * (np.log(CDF(y[i])) + np.log(1 - CDF(y[n - i - 1]))))
    ).sum()
    A2 = -n - S
    return A2


class MultiGaussian:
    """
    MultiGaussian
    an object like a scipy.stats distribution for a weighted sum of Gaussians

    z=MultiGaussian(means,stds,weights)

    methods:
    z.pdf(x)
    z.cdf(x)

    evaluate the pdf() and cdf() at x

    z.rvs(size)

    generate random variables according to the distribution of specified size

    """

    def __init__(self, means, stds, weights):
        """
        z=MultiGaussian(means,stds,weights)
        """

        self.M = len(means)
        self.weights = weights
        self.p = []
        for i in range(self.M):
            self.p.append(scipy.stats.norm(loc=means[i], scale=stds[i]))

    def pdf(self, x):
        """
        pdf(self, x)
        return probability distribution function evaluated at x
        """
        y = np.zeros_like(x)
        for i in range(self.M):
            y += self.weights[i] * self.p[i].pdf(x)
        return y

    def cdf(self, x):
        """
        cdf(self, x)
        return cumulative distribution function evaluated at x
        """

        y = np.zeros_like(x)
        for i in range(self.M):
            y += self.weights[i] * self.p[i].cdf(x)
        return y

    def rvs(self, size=(1,)):
        """
        rvs(self, size=(1,))
        generate random variables of specified size
        """
        w = scipy.stats.uniform.rvs(size=size)
        y = []
        for i in range(self.M):
            y.append(self.p[i].rvs(size=size))
        Y = y[0]
        for i in range(1, self.M):
            Y[w > self.weights[:i].sum()] = y[i][w > self.weights[:i].sum()]
        return Y


class NullingPulsar:
    """
    NullingPulsar()

    A class to do MCMC fitting of a 1-D Gaussian Mixture Models
    for nulling pulsars

    Example:

    import nulling_mcmc
    NP=nulling_mcmc.NullingPulsar(on, off, 2)
    means_fit, means_err, stds_fit, stds_err, weights_fit, weights_err, samples, lnprobs=NP.fit_mcmc(nwalkers=nwalkers,
                                                                                                 niter=niter,
                                                                                                 ninit=50,
                                                                                                 nthreads=nthreads,
                                                                                                 printinterval=50)

                                                                                                 
    """

    def __init__(self, on, off, M=2, component=gaussian):
        """
        __init__(on, off, M=2, component=gaussian)

        initialize the NullingPulsar object
        on and off are arrays with the intensities of pulses in the ON and OFF pulse windows
        should be of equal length
        M is the number of components (M=1 means no nulling, M=2 means standard nulling)
        component is a function that evaluates the distribution for all of the components (currently all must be the same).        
        it is run with component(x, [means, stds])
        """

        self.on = np.array(on)
        self.off = np.array(off)

        assert len(on) == len(off)

        # number of components
        self.M = M
        # functional form of each component
        if isinstance(component, list):
            if len(component) == M:
                self.component = component
            else:
                raise IndexError(
                    "Number of component functions must equal number of components"
                )
        else:
            self.component = [component] * M

        # only M-1 weights are independent
        self.n_parameters = 3 * self.M - 1
        # compute the parameters of the OFF distribution
        # want the best fit mean and width, along with uncertainties
        self.mean_off = self.off.mean()
        # self.std_off=self.off.std()
        # use the inner-quartile range for the standard deviation
        self.std_off = np.diff(np.percentile(self.off, [25, 75])) / 1.35
        self.mean_off_err = self.std_off / np.sqrt(len(self.off))
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.127.2352&rep=rep1&type=pdf
        self.std_off_err = self.std_off / np.sqrt(2 * (len(self.off) - 1))

        # number of points
        self.N = len(on)

        # a temporary place to compute probabilities
        self._p = np.zeros((self.N, self.M))

    def initfit(self):
        """
        means, stds, weights=initfit()
        do an initial fit using simple Gaussian mixture model
        to get a starting point

        requires sklean module

        """
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(self.M, tol=1e-4, init_params="random").fit(
            self.on.reshape(len(self.on), 1)
        )
        # and we need to sort by mean to make sure the NULL is in index 0
        means = model.means_[np.argsort(model.means_[:, 0]), 0]
        stds = np.sqrt(model.covariances_[np.argsort(model.means_[:, 0]), 0, 0])
        weights = model.weights_[np.argsort(model.means_[:, 0])]

        return means, stds, weights

    def loglike(self, means, stds, weights):
        """
        loglike(self, means, stds, weights)
        returns the natural log of the likelihood function
        
        means and stds should be arrays of length M
        weights is an array of length M
        See http://www.astroml.org/index.html
        Eqn. 4.18
        """
        for j in range(self.M):
            self._p[:, j] = weights[j] * self.component[j](self.on, [means[j], stds[j]])
        if np.any(self._p == 0):
            # worry about underflows
            self._p[self._p == 0] = 1e-99
        return np.log(self._p.sum(axis=1)).sum()

    def logprior(self, means, stds, weights):
        """
        logprior(self, means, stds, weights)
        returns the natural log of the prior distribution
        
        means and stds should be arrays of length M
        weights is an array of length M
        """
        if np.any(weights < 0) or np.any(weights > 1):
            return -np.inf
        if np.abs(weights.sum() - 1) > 1e-10:
            return -np.inf
        if np.any(stds < 0):
            return -np.inf
        if np.any(means < self.on.min()) or np.any(means > self.on.max()):
            return -np.inf
        # just the prior on the OFF distribution
        # a dirichlet prior on the weights is flat as long as the sum is 1
        # this should only be for when M>1: when M=1 we don't constrain this
        if self.M > 1:
            return float(
                -0.5 * ((means[0] - self.mean_off) / self.mean_off_err) ** 2
                - 0.5 * ((stds[0] - self.std_off) / self.std_off_err) ** 2
            )
        else:
            return 0

    def BIC(self, means, stds, weights):
        """
        BIC(self, means, stds, weights)        
        compute the Bayesian information criterion
        
        means and stds should be arrays of length M
        weights is an array of length M
        the lower the better
        """
        return -2 * (
            self.logprior(means, stds, weights) + self.loglike(means, stds, weights)
        ) + (3 * self.M - 1) * np.log(self.N)

    def AIC(self, means, stds, weights):
        """
        AIC(self, means, stds, weights)
        compute the Akaike information criterion

        means and stds should be arrays of length M
        weights is an array of length M
        the lower the better
        """
        return (
            -2
            * (self.logprior(means, stds, weights) + self.loglike(means, stds, weights))
            + (3 * self.M - 1) * 2
        )

    def pdf(self, means, stds, weights, x):
        """
        pdf(self, means, stds, weights, x):
        return probability distribution function of each component evaluated at x

        means and stds should be arrays of length M
        weights is an array of length M
        """
        pdf = np.zeros((self.M, len(x)))
        for j in range(self.M):
            pdf[j] = weights[j] * self.component[j](x, [means[j], stds[j]])
        return pdf

    def null_probabilities(self, means, stds, weights, x):
        """
        null_probabilities(self, means, stds, weights, x):
        returns the probabilities that a pulse with intensity x is from the null component

        means and stds should be arrays of length M
        weights is an array of length M        
        """
        pdfs = self.pdf(means, stds, weights, x)
        return pdfs[0] / float(pdfs.sum(axis=0))

    def __call__(self, theta):
        """
        __call__(self, theta)

        returns the log of the prior + the log of the likelihood
        
        means=theta[:M]
        stds=theta[M:2*M]
        weights=theta[2*M:]
        except there are M-1 weights passed here
        """
        means = theta[: self.M]
        stds = theta[self.M : 2 * self.M]
        weights = np.zeros_like(stds)
        weights[: self.M - 1] = theta[2 * self.M :]
        weights[-1] = 1 - weights.sum()
        prior = self.logprior(means, stds, weights)
        if np.isinf(prior):
            return -np.inf
        else:
            return prior + self.loglike(means, stds, weights)

    def fit_Richtings(self):
        """
        nf=fit_Richtings()
        fit for nulling fraction using the method of Richtings (1976)
        construct histograms of ON and OFF
        find the nulling fraction such that the sum of bins with intensity<0
        for ON-nf*OFF is closest to 0
        """
        nf = np.linspace(0, 1, 200) + np.random.uniform(0, 0.01)
        delta = np.zeros(len(nf))
        x = np.linspace(self.on.min(), self.on.max(), 100)
        onhist = np.histogram(self.on, x)[0]
        offhist = np.histogram(self.off, x)[0]
        for i in range(len(nf)):
            y = onhist - offhist * nf[i]
            delta[i] = y[x[:-1] < 0].sum()

        return nf[np.abs(delta) == np.abs(delta).min()][0]

    def fit_mcmc(
        self,
        means=None,
        stds=None,
        weights=None,
        nwalkers=40,
        niter=500,
        ninit=50,
        nthreads=4,
    ):
        """
        means_fit, means_err, stds_fit, stds_err, weights_fit, weights_err, samples, lnprobs=fit_mcmc(
        means=None, stds=None, weights=None,
                                  nwalkers=40, niter=500, ninit=50, nthreads=4)
        means, stds, weights are the initial values to use for the fits
        if any of these is None, then computes starting values using the scikit-learn GMM routine
        
        nwalkers is number of walkers
        niter is number of fit iterations (total chains = nwalkers * niter)
        ninit is number of iterations for burn-in
        nthreads is for multi-threading

        """
        ndim = self.n_parameters
        # first, fit using plain GMM to get an initial value
        if means is None or stds is None or weights is None:
            means, stds, weights = self.initfit()

        init_means = np.random.normal(means, (stds), size=(nwalkers, self.M))
        init_stds = scipy.stats.truncnorm.rvs(
            (0 - stds) / stds,
            (self.on.max() - stds) / stds / 2.0,
            loc=stds / 2,
            scale=stds / 2.0,
            size=(nwalkers, self.M),
        )
        init_weights = scipy.stats.dirichlet.rvs(np.ones(self.M), size=(nwalkers))
        # only pass M-1 independent weights
        p0 = np.hstack((init_means, init_stds, init_weights[:, :-1]))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self, threads=nthreads)

        sampler.run_mcmc(p0, niter, progress=True)

        # reject ninit samples from burn-in
        samples = sampler.get_chain(discard=ninit, flat=True)
        lnprobs = sampler.get_log_prob(discard=ninit, flat=True)

        means_fit = np.median(samples[:, : self.M], axis=0)
        stds_fit = np.median(samples[:, self.M : 2 * self.M], axis=0)
        means_err = np.std(samples[:, : self.M], axis=0)
        stds_err = np.std(samples[:, self.M : 2 * self.M], axis=0)
        weights_fit = np.zeros(self.M)
        weights_fit[: self.M - 1] = np.median(samples[:, 2 * self.M :], axis=0)
        weights_fit[self.M - 1] = 1 - weights_fit.sum()
        weights_err = np.zeros(self.M)
        weights_err[: self.M - 1] = np.std(samples[:, 2 * self.M :], axis=0)
        weights_err[self.M - 1] = (1 - samples[:, 2 * self.M :].sum(axis=1)).std(axis=0)

        del sampler

        return (
            means_fit,
            means_err,
            stds_fit,
            stds_err,
            weights_fit,
            weights_err,
            samples,
            lnprobs,
        )
