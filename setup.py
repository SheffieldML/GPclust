import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "GPclust",
    version = "1.0.0",
    author = "James Hensman",
    author_email = "james.hensman@sheffield.ac.uk",
    description = ("Clustering of time series using Gaussian Processes and Variational Bayes"),
    license = "GPL v3",
    keywords = " clustering Gaussian-process machine-learning",
    url = "http://jameshensman.githubio.GPclust",
    packages=['src'],
    long_description=read('README'),
    classifiers=[
        "Topic :: machine learning",
        "License :: OSI Approved :: GPLv3 License",
    ],
)
