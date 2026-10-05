library(ks)
library(LaplacesDemon)

### MLA for Poisson Graphical model ###


### Function to generate data from a PGM with parameter theta0 ###

PGMsim = function(n, theta, max_iter)
{
  d = ncol(theta)
  X = matrix(rpois(n*d, exp(diag(theta))), n, d)
  for(k in 1:max_iter)
  {
    for(j in 1:d)
    {
      rate_j = exp(theta[j,j] + 2*X[, -j]%*%matrix(theta[-j, j], d - 1, 1))
      X[,j] = rpois(n, rate_j)
    }
  }
  return(X)
}

### Unbiased estimation of z(theta) for PGM for a given theta ###
ztheta = function(theta, N)
{
  # N = number of importance samples to be drawn from 
  phi = diag(theta)
  d = nrow(theta)
  Y = matrix(0, N, d)
  for(j in 1:d)
  {
    Y[,j] = rpois(N, exp(phi[j]))
  }
  s = 0
  for(i in 1:N)
  {
    s = s + exp(t(Y[i,])%*%(theta-diag(phi))%*%Y[i,])
  }
  z_phi = 1
  return(z_phi*(s/N))
}

### Unbiased estimation of log z(theta) for PGM for a given theta ###
log_ztheta = function(theta, N, p0)
{
  R = rgeom(1,p0) + 1
  a = ztheta(theta, N)
  s = log(a)
  #print(R)
  for(k in 1:R)
  {
    T_vec = replicate(k, ztheta(theta,N))
    
    s = s + ((-1)^(k-1)/k)*prod((T_vec - rep(a, k))/rep(a, k))/((1-p0)^(k-1))
    
    #print(s)
  }
  
  return(s)
}


### Unbiased estimator of negative log-posterior for a given data and given theta ###
### The parameter theta_vec is the half-vectorization of the symmetric d by d matrix theta ###
### i.e. theta_vec is a d(d+1)/2 vector. We assume a product prior on theta where the diagonal elements ###
### have a Normal prior with mean 0 and the off-diagonal elements have a truncated Normal prior with mean 0 ###
### X = data matrix of size n by d ###
### N = number of importance samples to draw from p_\phi to produce an estimate of z(\theta)/z(\phi). Default = 1000. ###
### p0 = the parameter of the Geometric distribution that is used to construct the unbiased estimator of \log z(\theta)/z(\phi). Default = 0.7 ###
### prior_sd = sd of the Gaussian prior on the elements of theta. Default = 10.

unbiased_log_posterior = function(theta_vec, X, N = 1000, M = 10, p0 = 0.99, prior_sd = 10)
{
  n = nrow(X)
  d = ncol(X)
  theta = invvech(theta_vec)
  # print(theta)
  S = t(X)%*%X
  diag(S) = apply(X, 2, sum)
  t1 = sum(diag(S%*%theta))
  t2 = n*log_ztheta(theta, N, p0) + n*(sum(exp(diag(theta)))) # log(z(theta)/z(phi)) + log(z(phi))
  # t2 = n*log(ztheta(theta, N)) + n*prod(exp(diag(theta)))
  t3 = sum(dnorm(diag(theta), 0, prior_sd, log = T)) + sum(log(dtrunc(theta[lower.tri(theta)], "norm", a = -Inf, b = 0, mean = 0, sd = prior_sd)))
  # print(c(t1,t2,t3, log_ztheta(theta, N, p0), prod(exp(diag(theta)))))
  # print(log(ztheta(theta,N)))
  return(-t1 + t2 - t3)
}

biased_log_posterior = function(theta_vec, X, N = 1000, p0 = 0.7, prior_sd = 10)
{
  n = nrow(X)
  d = ncol(X)
  theta = invvech(theta_vec)
  S = t(X)%*%X
  diag(S) = apply(X, 2, sum)
  #print(S)
  t1 = sum(diag(S%*%theta))
  #print(t1)
  t2 = n*log(ztheta(theta, N)) + n*(sum(exp(diag(theta)))) # It was previously exp(sum(diag(theta))) or prod(exp(diag(theta)))
  #print(t2)
  # t
  t3 = sum(dnorm(diag(theta), 0, prior_sd, log = T)) + sum(log(dtrunc(theta[lower.tri(theta)], "norm", a = -Inf, b = 0, mean = 0, sd = prior_sd)))
  return((-t1 +t2 - t3))
}

# 
# lg_post_for_gradient_estimate = function(theta_vec, X, N = 1000, p0 = 0.7, prior_sd = 10, lg_ztheta)
# {
#   n = nrow(X)
#   d = ncol(X)
#   theta = invvech(theta_vec)
#   #print(theta)
#   S = t(X)%*%X
#   diag(S) = apply(X, 2, sum)
#   t1 = sum(diag(S%*%theta))
#   t2 = n*lg_ztheta + n*prod(exp(diag(theta)))
#   t3 = sum(dnorm(diag(theta), 0, prior_sd, log = T)) + sum(log(dtrunc(theta[lower.tri(theta)], "norm", a = -Inf, b = 0, mean = 0, sd = prior_sd)))
#   return(-t1 + t2 - t3)
# }


pseudo_likelihood = function(X, penalty, lambda)
{
  p = ncol(X)
  theta_star= matrix(0, p, p)
  if(penalty == "none"){
    for(j in 1:p)
    {
      X1 = 2*X[,-j]
      res = glm(X[,j] ~ X1, family = 'poisson')$coefficients
      theta_star[j,j] = res[1]
      theta_star[j, -j] = res[-1]
    }
    theta_star = (theta_star + t(theta_star))/2
    return(theta_star)
  }else if(penalty == "l1"){
    for(j in 1:p)
    {
      X1 = 2*X[,-j]
      res1 = cv.glmnet(X1, X[,j], family = 'poisson', alpha = 1)
      res2 = glmnet(X1, X[,j], family = 'poisson', lambda = res1$lambda[res1$index[1]])
      #res2 = glmnet(X1, X[,j], family = 'poisson', lambda = lambda)
      b = as.matrix(coef(res2))
      theta_star[j,j] = b[1]
      theta_star[-j, j] = b[-1]
    }
    theta_star = (theta_star + t(theta_star))/2
    return(theta_star)
  }
}
