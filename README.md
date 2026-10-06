# sFO-MLA

R and Rcpp implementations of the stochastic first-order Mirror Langevin Algorithm (sFO-MLA) and its warm-started two-loop variant for sampling from constrained distributions.

The code accompanies the manuscript **"Two-Loop Stochastic Mirror Langevin Algorithms for Constrained Sampling."** It includes implementations and experiment scripts for Bayesian mixture-weight inference on the simplex and posterior sampling in Poisson graphical models with an intractable normalizing constant.

## Description

Mirror Langevin algorithms map a constrained sampling problem to an unconstrained dual space through a mirror map. Standard fixed-step implementations mix quickly at larger step sizes but retain a nonvanishing discretization error, while smaller step sizes reduce this error at the cost of slower mixing.

The two-loop sFO-MLA implementation uses a geometric step-size schedule. At each outer epoch, the algorithm runs a fixed-step sFO-MLA chain for a corresponding number of inner iterations and warm-starts the next epoch from the current endpoint. The accompanying paper establishes finite-time Wasserstein bounds separating transient mixing, Euler-Maruyama discretization, and stochastic-gradient errors, and shows that the two-loop construction attains an \(O(T^{-1/2})\) rate under the stated assumptions.

The repository contains two applications:

- **Bayesian mixture weights:** sampling mixture weights on a simplex, with stochastic gradients obtained by minibatching observations.
- **Poisson graphical models:** sampling a constrained posterior with an intractable normalizing constant, with stochastic gradients estimated by Monte Carlo simulation.

## Installation

Clone the repository:

```bash
git clone https://github.com/RuitingDeposit/sFO-MLA.git
cd sFO-MLA
```

Install the required R packages:

```r
install.packages(c(
  "Rcpp",
  "RcppArmadillo",
  "nleqslv",
  "ks",
  "LaplacesDemon",
  "ggplot2",
  "latex2exp",
  "T4transport",
  "rmarkdown",
  "knitr"
))
```

The Rcpp implementations are compiled when `Rcpp::sourceCpp()` is called, so a working C++ toolchain is also required. This repository contains research scripts rather than an installable R package.

## Examples

### Bayesian mixture weights on the simplex

The following example simulates observations from a three-component Gaussian mixture and runs the Rcpp implementation of two-loop sFO-MLA. Run it from the repository root.

```r
library(Rcpp)

sourceCpp("mixture_model_code/sFO_MLA_mixture_model.cpp")

set.seed(1)

n <- 500L
n_components <- 3L
alpha <- rep(2, n_components)
true_weights <- c(0.2, 0.3, 0.5)

classes <- sample.int(
  n_components,
  size = n,
  replace = TRUE,
  prob = true_weights
)
observations <- rnorm(n, mean = classes, sd = 0.2)

# The first K - 1 mixture weights are free. The constraints are
# theta_j >= 0 and sum(theta_1, ..., theta_{K-1}) <= 1.
A <- rbind(
  -diag(n_components - 1),
  rep(1, n_components - 1)
)
b <- c(rep(0, n_components - 1), 1)

fit <- DMLA_new_grad_cpp_mix_lglik(
  theta0 = rep(1 / n_components, n_components - 1),
  x = observations,
  alpha = alpha,
  B = 20L,
  eta0 = 0.05,
  rho = 0.96,
  lam = 1,
  multiplier = 0.05,
  A = A,
  b = b,
  scale = n,
  message = TRUE,
  m = 100L,
  std = 0.2
)

last_free_weights <- fit$primal_samples[, ncol(fit$primal_samples)]
estimated_weights <- c(
  last_free_weights,
  1 - sum(last_free_weights)
)
estimated_weights
```

The returned list includes:

- `primal_samples`: samples in the constrained primal space at the outer epochs;
- `dual_samples`: corresponding samples in the unconstrained dual space;
- `all_x`: all inner-loop primal samples;
- `outer_loop_indices`: locations of outer-epoch endpoints among the inner samples;
- `all_log_liklihood`: log-posterior values recorded during sampling; and
- `time`: elapsed runtime.

An R-only reference implementation and a longer demonstration are available in `man/sMLA.R` and `demo/sFO_MLA_mixed_dist.Rmd`, respectively.

### Poisson graphical model

The Poisson graphical model implementation combines R helper functions with Rcpp/RcppArmadillo routines. Run the following setup from the repository root:

```r
setwd("PGM_code")

source("PGM.R")
source("PGM_sFO_MLA_functions.R")
```

The complete data-generation, two-loop sampling, exchange-algorithm comparison, and plotting workflow is provided in `PGM_sFO_MLA.Rmd`. The manuscript experiments use \(p=4\) and \(p=5\), with 500 observations generated from a Poisson graphical model whose diagonal parameters are 1 and whose off-diagonal parameters are -0.1.

## Reproducing the experiments

The main experiment files are:

```text
demo/
  sFO_MLA_mixed_dist.Rmd
man/
  sMLA.R
mixture_model_code/
  sFO_MLA_mixture_model.cpp
  sFO_MLA_mixed_dist_main_experiments.Rmd
  sFO_MLA_mixed_dist_Plots.Rmd
  sFO_mixed_dist_test_minibatch_effect.Rmd
PGM_code/
  PGM.R
  PGM_sFO_MLA.cpp
  PGM_sFO_MLA_functions.R
  PGM_sFO_MLA.Rmd
```

The mixture-model experiments consider \(K \in \{30, 50, 80, 100\}\), use \(n=10{,}000\) observations, a minibatch size of 2,500, geometric decay factor `rho = 0.96`, and 135 outer samples. The PGM experiments use Monte Carlo estimates of the intractable likelihood contribution and compare two-loop sFO-MLA with an exchange-algorithm correction.

The experiment notebooks use paths relative to their own directories. Set the working directory to `mixture_model_code/` or `PGM_code/` before running their chunks. Some plotting chunks load previously generated `.RDS` files that are not included in the repository; run the corresponding sampling and data-generation chunks first, create the referenced output directories, and then run the plotting chunks.

## Reference

*Two-Loop Stochastic Mirror Langevin Algorithms for Constrained Sampling.* Manuscript under review, 2026.

If you use this code, please cite the paper. Full bibliographic information will be added when the manuscript becomes publicly available.
