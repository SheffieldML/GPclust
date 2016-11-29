GPclust
=====

Clustering time series using Gaussian processes and variational Bayes. 

User guide and tutorials are available via the included [notebooks](https://github.com/mathDR/GPclust/tree/gpflowbase/notebooks/index.ipynb). 

Currently implemented models are

* MOG - Mixture of Gaussians
* MOHGP - Mixtures of Hierarchical Gaussian processes
* OMGP - Overlapping mixtures of Gaussian processes

Citation
========

The underlying algorithm is based on the 2012 NIPS paper:


http://books.nips.cc/papers/files/nips25/NIPS2012_1314.pdf
```TeX
@article{hensman2012fast,
  title={Fast variational inference in the conjugate exponential family},
  author={Hensman, James and Rattray, Magnus and Lawrence, Neil D},
  journal={Advances in Neural Information Processing Systems},
  year={2012}
}
```

The code also implements clustering of Hierachical Gaussian Processes using that inference framework, detailed in the two following works:

http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6802369
```TeX
@article{hensman2014fast,
  author={Hensman, J. and Rattray, M. and Lawrence, N.},
  journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
  title={Fast nonparametric clustering of structured time-series},
  year={2014},
  volume={PP},
  number={99},
  keywords={Biological system modeling;Computational modeling;Data models;Gaussian processes;Optimization;Time series analysis},
  doi={10.1109/TPAMI.2014.2318711},
  ISSN={0162-8828}
}
```

http://www.biomedcentral.com/1471-2105/14/252
```TeX
@article{hensman2013hierarchical,
  title={Hierarchical Bayesian modelling of gene expression time series across irregularly sampled replicates and clusters},
  author={Hensman, James and Lawrence, Neil D and Rattray, Magnus},
  journal={BMC bioinformatics},
  volume={14},
  number={1},
  pages={1--12},
  year={2013},
  publisher={BioMed Central}
}
```


Additionally Overlapping Mixtures of Gaussian Processes model is implemented (using the variational methods described in the above), which was published in this paper:

```TeX
@article{Lazaro-Gredilla2012,
  title = {{Overlapping Mixtures of Gaussian Processes for the data association problem}},
  author = {L{\'{a}}zaro-Gredilla, Miguel and {Van Vaerenbergh}, Steven and Lawrence, Neil D.},
  doi = {10.1016/j.patcog.2011.10.004},
  journal = {Pattern Recognition},
  month = {apr},
  number = {4},
  pages = {1386--1395},
  url = {},
  volume = {45},
  year = {2012}
}
```



Dependencies
------------

This work depends on the [GPflow project](https://github.com/GPflow), as well as the 
numpy/scipy stack. matplotlib is optional for plotting. 


Contributors
------------

- James Hensman
- Valentine Svensson
- Max Zwiessele
- Daniel Marthaler
