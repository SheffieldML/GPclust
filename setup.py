from setuptools import setup

setup(
    name = "GPclust",
    version = "0.1.0",
    author = "James Hensman",
    author_email = "james.hensman@sheffield.ac.uk",
    url = "http://staffwww.dcs.sheffield.ac.uk/people/J.Hensman/gpclust.html",
    description = ("Clustering of time series using Gaussian processes and variational Bayes"),
    license = "GPL v3",
    keywords = " clustering Gaussian-process machine-learning",
    download_url = 'https://github.com/jameshensman/gpclust/tarball/0.1',
    packages=['GPclust'],
    install_requires=['GPflow>=0.4.0'],
    classifiers=[]
)
