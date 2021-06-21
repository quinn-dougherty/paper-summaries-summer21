# Gaussian Error Linear Units (GELUs) - paper summary

This paper represents a simple idea. Traditional `ReLU` activation is known to perform quite well and is expressed `\x -> max(x, 0)` or `x` times an indicator function which is `1` when `x > 0` and `0` otherwise. The `GELU` or Gaussian Error Linear Unit is simply replacing the the indicator function with `P(X <= x), X ~ N(0, 1)` (the probability that `X` is less than or equal to `x` with `X` drawn from the normal distribution centered at `0` with stdev `1`). 

The bulk of the result is how `GELU` compares to `ReLU`. They also compare it to `ELU`, the activation which occasionally allows negative values in the output (which speeds up training). There is also potential for a `SiLU` or `sigmoid linear unit` which is similar to `GELU` because `sigmoid` approximates the normal distribution's CDF, but it ultimately underperforms when compared to `GELU`. 

With these activations, the authors show that `GELU` outperforms `ReLU` on various tasks such as MNIST classification, CIFAR classification, and others. 

## relation to dropout

If you think about it, as values of `x` decrease they get a higher probability of being "dropped", so it's sort of like adaptive dropout. 

Moreover, the performance increase offered by `GELU` on MNIST is made much more drastic in the dropout (0.5) case, as figure 2 shows. 
