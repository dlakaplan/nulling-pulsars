"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='nulling_mcmc',

    version='0.9',

    description='MCMC fitting of Gaussian Mixture Model for nulling pulsars',

    # The project's main homepage.
    url='https://github.com/dlakaplan/nulling-pulsars',

    # Author details
    author='David Kaplan',
    author_email='kaplan@uwm.edu',

    # Choose your license
    license='TBD',


    py_modules=["nulling_mcmc"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy','scipy','emcee'],
)
