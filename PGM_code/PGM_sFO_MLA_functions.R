library(Rcpp)
library(RcppArmadillo)
Rcpp::sourceCpp("PGM_sFO_MLA.cpp")

not_diag = function(dX){
  if(dX == 1){return(0)}
  else{
    v = numeric(0)
    for(i in (dX-1):0){
      temp = c(0, rep(1,i))
      v = c(v,temp)
    }
    return(v)
  }
}

# DMLA with 1st order + log likelihood
DMLA_order1_cpp_lglik <- function(
    grad,
    x0,
    X,
    K,
    eta0,
    rho,
    lam,
    multiplier,
    A,
    b,
    scale = 1,
    message = FALSE,
    ...
) {
  d <- length(x0)
  
  non_diag <- not_diag(floor(sqrt(2 * d)))
  
  DMLA_new_grad_cpp_impl_lglik(
    grad = grad,
    x0 = x0,
    X = X,
    K = K,
    eta0 = eta0,
    rho = rho,
    lam = lam,
    multiplier = multiplier,
    A = A,
    b = b,
    scale = scale,
    message = message,
    non_diag_r = non_diag,
    dots = list(...)
  )
}

# 0th order + log likelihood
DMLA_order0_cpp_lglik <- function(
    Fn,
    x0,
    dat,
    K,
    eta0,
    rho,
    lam,
    grad_method,
    s0,
    multiplier,
    A,
    b,
    scale = 1,
    message = FALSE,
    MH = F,
    ...
) {
  d <- length(x0)
  
  non_diag <- not_diag(floor(sqrt(2 * d)))
  
  DMLA_new_cpp_impl_lglik(
    Fn = Fn,
    x0 = x0,
    dat = dat,
    K = K,
    eta0 = eta0,
    rho = rho,
    lam = lam,
    grad_method = grad_method,
    s0 = s0,
    multiplier = multiplier,
    A = A,
    b = b,
    scale = scale,
    message = message,
    non_diag_r = non_diag,
    MH = MH,
    dots = list(...)
  )
}

DMLA_order1_MH <- function(
    grad,
    x0,
    X,
    K,
    eta0,
    lam,
    A,
    b,
    scale = 1,
    message = FALSE,
    ...
) {
  d <- length(x0)
  
  non_diag <- not_diag(floor(sqrt(2 * d)))
  
  DMLA_PGG_MH_order1(
    grad = grad,
    x0 = x0,
    X = X,
    K = K,
    eta0 = eta0,
    lam = lam,
    A = A,
    b = b,
    scale = scale,
    message = message,
    non_diag_r = non_diag,
    dots = list(...)
  )
}

## 0th order MH
DMLA_order0_MH <- function(
    Fn,
    x0,
    X,
    K,
    eta0,
    lam,
    grad_method,
    s0,
    A,
    b,
    scale = 1,
    message = FALSE,
    ...
) {
  d <- length(x0)
  
  non_diag <- not_diag(floor(sqrt(2 * d)))
  
  DMLA_PGG_MH_order0(
    Fn = Fn,
    x0 = x0,
    X = X,
    K = K,
    eta0 = eta0,
    lam = lam,
    grad_method = grad_method,
    s0 = s0,
    A = A,
    b = b,
    scale = scale,
    message = message,
    non_diag_r = non_diag,
    dots = list(...)
  )
}

# 1st_order_one_loop
DMLA_order1_one_loop <- function(
    grad,
    x0,
    X,
    K,
    eta0,
    lam,
    multiplier,
    A,
    b,
    scale = 1,
    message = FALSE,
    ...
) {
  d <- length(x0)
  
  non_diag <- not_diag(floor(sqrt(2 * d)))
  
  DMLA_new_grad_cpp_one_loop(
    grad = grad,
    x0 = x0,
    X = X,
    K = K,
    eta0 = eta0,
    lam = lam,
    multiplier = multiplier,
    A = A,
    b = b,
    scale = scale,
    message = message,
    non_diag_r = non_diag,
    dots = list(...)
  )
}