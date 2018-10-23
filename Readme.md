## Sparips.jl: Practical Sparsification of Rips-complexes for approximating persistent homology

`Sparips.jl` provides a sparsifying preprocessor for the computation of persistent homology of Rips complexes of finite metric spaces. The original paper and algorithm is available on the [arxiv](https://arxiv.org/abs/1807.09982). If you use `Sparips.jl` in your research, please cite the paper, as well as Ulrich Bauer who wrote the underlying solver [ripser](https://github.com/Ripser/ripser). 

The usage of `Sparips.jl` is explained in the [tutorial](./docs/tutorial/sparips_tutorial.md). 

We have some preliminary [benchmarks](./docs/benchmarks/comparison_gudhi.md) of sparsification rates and runtimes, compared to [gudhi](http://gudhi.gforge.inria.fr/).

The API has not yet stabilized; it is for this reason that the documentation is still rather lacking. If there are components of `Sparips.jl` that you would like to use in a different project, please open an issue. For example, the metric contraction tree might be a useful basis also in the context of manifold learning.


## Installation

`Sparips.jl` is not registered yet. It can be installed by 

```
(v1.0) pkg> add https://github.com/bbrehm/Sparips.jl
(v1.0) pkg> build Sparips
```
