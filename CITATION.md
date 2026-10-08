# Citation

If you use this code, please cite the manuscript:

> Liu, Z., He, Y., Ding, M., Yuan, Z., Zhou, X., Huang, J., Wu, X., Lu, X.,
> Tao, J., Lee, T. M. C., Chen, L. & Wu, J. Network-derived clinical indicator of
> symptom system stability and core symptoms in depression: a cross-cultural
> validation study. *Nature Mental Health* (in submission).

BibTeX:

```bibtex
@article{cesd_ising_snci,
  title   = {Network-derived clinical indicator of symptom system stability and
             core symptoms in depression: a cross-cultural validation study},
  author  = {Liu, Zhihan and He, Youze and Ding, Minjiang and Yuan, Zhiwei and
             Zhou, Xinye and Huang, Jinshan and Wu, Xinyu and Lu, Xiuxu and
             Tao, Jing and Lee, Tatia M. C. and Chen, Lidian and Wu, Jingsong},
  journal = {Nature Mental Health},
  year    = {2026},
  note    = {Code available at https://github.com/zhihan-liu023/CESD-Ising-Network}
}
```

## Methodological references

The implementation follows the conventions of the works cited in the
manuscript. The mapping from each numerical routine to its source is:

- Ising model estimation and network accuracy:
  Epskamp, S., Borsboom, D. & Fried, E. I. (2018). Estimating psychological
  networks and their accuracy: a tutorial paper. *Behavior Research Methods* 50,
  195–212.
- Constructing networks from binary data:
  van Borkulo, C. D. et al. (2014). A new method for constructing networks from
  binary data. *Scientific Reports* 4, 5918.
- Interpretation and scaling of the Ising model:
  Haslbeck, J. M. B., Epskamp, S., Marsman, M. & Waldorp, L. J. (2021).
  Interpreting the Ising model: the input matters. *Multivariate Behavioral
  Research* 56, 303–313.
- Reporting standards for psychological network analyses:
  Burger, J. et al. (2023). Reporting standards for psychological network
  analyses in cross-sectional data. *Psychological Methods* 28, 806–817.
- Extended Bayesian information criterion:
  Foygel, R. & Drton, M. (2010). Extended Bayesian information criteria for
  Gaussian graphical models. *Advances in Neural Information Processing
  Systems* 23, 604–612.
- Percolation theory:
  Dietrich, S. (1992). *Introduction to Percolation Theory*.
- Percolation analysis of a connectome (the application this study transplants
  into symptom networks):
  Kotlarz, P., Nino, J. C. & Febo, M. (2022). Connectomic analysis of
  Alzheimer's disease using percolation theory. *Network Neuroscience* 6,
  213–233.
- Energy landscape analysis:
  Ezaki, T., Watanabe, T., Ohzeki, M. & Masuda, N. (2017). Energy landscape
  analysis of neuroimaging data. *Philosophical Transactions of the Royal
  Society A* 375, 20160287.
- Louvain community detection:
  Blondel, V. D., Guillaume, J. L., Lambiotte, R. & Lefebvre, E. (2008). Fast
  unfolding of communities in large networks. *Journal of Statistical Mechanics:
  Theory and Experiment* P10008.
- Modularity:
  Newman, M. E. J. (2006). Modularity and community structure in networks.
  *Proceedings of the National Academy of Sciences* 103, 8577–8582.

The network comparison test implemented in `src/cross_cohort_analysis.py` (disable with
`--no-nct`) follows van Borkulo, C. D. et al. (2022). Comparing network structures on three
aspects: a permutation test. *Psychological Methods* 28, 1273–1285. That test is
exploratory and is **not** reported in the manuscript.
