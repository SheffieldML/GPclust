colvb
=====

Collapsed Variational Bayes. 

***

Currently implemented models include

* LDA - Latent Dirichlet Allocation
* MOG - Mixture of Gausians
* MMOG - Multiple MOG (which is kind of like a continuous version of LDA)
* MOHGP - Mixtures of Hierarchical Gaussian Processes

Other files are common to several models:
col_vb contains a skeleton for the class, and includes the optimiser
col_mix contains the base class for a mixture model (MOG, MOHGP)

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

http://arxiv.org/abs/1401.1605
```TeX
@article{hensman2014fast,
  title={Fast variational inference for nonparametric clustering of structured time-series},
  author={Hensman, James and Rattray, Magnus and Lawrence, Neil D},
  journal={arXiv preprint arXiv:1401.1605},
  year={2014}
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




Dependencies
------------

This work depends on some functionality from the [GPy project](https://github.com/SheffieldML/GPy), as well as the numpy/scipy/matplotlib stack.

I've tested the demos with GPy v0.2, but it should work with later versions also. 
