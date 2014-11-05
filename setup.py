import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "GPclust",
    version = "0.1.0",
    author = "James Hensman",
    author_email = "james.hensman@sheffield.ac.uk",
    url = "http://staffwww.dcs.sheffield.ac.uk/people/J.Hensman/gpclust.html",
    description = ("Clustering of time series using Gaussian processes and variational Bayes"),
    license = "GPL v3",
    keywords = " clustering Gaussian-process machine-learning",
    packages=['GPclust'],
    long_description=read('README'),
    classifiers=[
        "Topic :: machine learning",
        "License :: OSI Approved :: GPLv3 License",
    ],
)
