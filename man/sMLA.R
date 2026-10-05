library(nleqslv)
##
## Fn: zeroth order oracle
## x0: initial point
## K: # iterations in the outer loop 
## eta0: initial step size in the Euler Maruyama algorithm
## rho: geometric decay of eta
## lam: the regularizer in the log barrier mirror map
## grad_method: forward / center; defines the manner of computing the gradient
## s0: the s in the gradient calculation is s0*eta^power, power = 1/4 if grad_method = forward and 1/6 if grad_method = center.
## multiplier: multiplies the number of inner loops by this factor
## A, b: linear constraints
## scale: scale the gradient down by this factor
## message: print iteration + sample
##
sMLA_new = function(Fn, x0, K, eta0, rho, lam, grad_method, s0, multiplier, A, b, scale, message, ...){
  #phi
  d = length(x0)
  phi = function(x, lam){
    r = -sum(log(-(A%*%x - b))) + lam*sum(x^2)/2
    return(r)
  }
  
  # Nabla phi
  nabla_phi = function(x,lam){
    r1 = A%*%x - b
    r = -t(A)%*%(1/r1) + lam*x
    return(r)
  }
  
  # Hessian
  Hess = function(x, lam){
    r1 = A%*%x - b
    D = diag(1/as.vector(r1)^2)
    H = t(A)%*%D%*%A + lam*diag(length(x))
    return(H)
  }
  
  # Hessian^0.5
  sHess = function(x, lam){
    decomp = eigen(Hess(x, lam))
    V = decomp$vectors
    L = decomp$values
    return(V %*% diag(sqrt(L)) %*% t(V))
  }
  
  # Check if the constraint is satisfied.
  #check_constraint = function(x, dX){
  #  return(all(x[not_diag(dX)]<0))
  #}  
  
  
  # Inverse of nabla phi
  inv_nabla_phi = function(x0,y,lam){
    g =function(x){
      nabla_phi(x, lam) - y
    }
    x = nleqslv(x0, g)$x
    return(x)
  }
  
  # inv_nabla_phi = function(y,lam){
  #   z = numeric(d)
  #   non_diag = not_diag(floor(sqrt(2*d)))
  #   for(i in 1:d){
  #     if(non_diag[i]){
  #       z[i] = (y[i] - sqrt(y[i]^2 + 4*lam))/(2*lam)
  #     }else{
  #       z[i] = y[i]/lam
  #     }
  #   }
  #   return(z)
  # }
  
  # Estimate gradient
  grad_est_centered = function(Fn, x, eps, ...){
    est = numeric(d)
    for(i in 1:d){
      ei = numeric(d)
      ei[i] = 1
      x1 = inv_nabla_phi(x0 = x, nabla_phi(x, lam) - ei*eps, lam)
      x2 = inv_nabla_phi(x0 = x, nabla_phi(x, lam) + ei*eps, lam)
      F1 = Fn(x1, ...)
      F2 = Fn(x2, ...)
      est[i] = (F2 - F1)/(2*eps)
    }
    return(est)
  }
  
  grad_est_forward = function(Fn, x, eps, ...){
    est = numeric(d)
    F0 = Fn(x, ...)
    for(i in 1:d){
      ei = numeric(d)
      ei[i] = 1
      x1 = inv_nabla_phi(x0 = x, nabla_phi(x, lam) + ei*eps, lam)
      F1 = Fn(x1, ...)
      est[i] = (F1-F0)/eps
    }
    return(est)
  }
  
  
  ## Choose which gradient estimate to use
  grad_est = NULL
  
  if(grad_method == "center"){
    grad_est = grad_est_centered
    power = 1/6
  }else{
    grad_est = grad_est_forward
    power = 1/4
  }
  
  eta = eta0
  
  #if(length(x0) != d){
  #  print("Length of x0 is incorrect for data X.")
  #  return()
  #}
  
  xs = matrix(numeric(K*d), nrow = d)
  ys = matrix(numeric(K*d), nrow = d)
  all_x = matrix(ncol = 0, nrow = d)
  outer_loop_index = numeric(K)
  
  
  xs[,1] = x0
  ys[,1] = nabla_phi(x0, lam)
  outer_loop_index[1] = 1
  
  t1 = Sys.time()
  for(i in 2:K){
    ti = ceiling(multiplier/eta)
    inner_xs = matrix(ncol = ti, nrow = d)
    inner_ys = matrix(ncol = ti, nrow = d) 
    inner_xs[,1] = xs[,i-1]
    inner_ys[,1] = ys[,i-1]
    eps = s0*eta^power
    
    #if(i %% 10 == 0){print(i)}
    
    if(ti >= 2){
      for (j in 2:ti){
        noi = rnorm(d,0,1)
        inner_ys[,j] = inner_ys[,j-1] - eta*grad_est(Fn, inner_xs[,j-1], eps, ...)/scale + 
          sqrt(2*eta/scale)*sHess(inner_xs[,j-1], lam)%*%noi
        inner_xs[,j] = inv_nabla_phi(x0 = inner_xs[,j-1], inner_ys[,j], lam)
      }
    }
    
    xs[,i] = inner_xs[,ti]
    ys[,i] = inner_ys[,ti]
    all_x = cbind(all_x, inner_xs)
    outer_loop_index[i] = outer_loop_index[i-1] + ti - 1
    eta = rho*eta
    
    if(message){
      print(paste("iteration:", i ," sample:", paste(xs[,i], collapse = " ")))
    }
  }
  t2 = Sys.time() - t1
  return(list("primal_samples" = xs, "dual_samples" = ys, "time" = t2, "all_x" = all_x, "outer_loop_indices" = outer_loop_index))
}


##
## uses gradient
##
##
sMLA_new_grad = function(grad, x0, K, eta0, rho, lam, multiplier, A, b, scale, message, ...){
  #phi
  d = length(x0)
  phi = function(x, lam){
    r = -sum(log(-(A%*%x - b))) + lam*sum(x^2)/2
    return(r)
  }
  
  # Nabla phi
  nabla_phi = function(x,lam){
    r1 = A%*%x - b
    #print(r1)
    r = -t(A)%*%(1/r1) + lam*x
    #print(r)
    return(r)
  }
  
  # Hessian
  Hess = function(x, lam){
    r1 = A%*%x - b
    D = diag(1/as.vector(r1)^2)
    H = t(A)%*%D%*%A + lam*diag(length(x))
    return(H)
  }
  
  # Hessian^0.5
  sHess = function(x, lam){
    decomp = eigen(Hess(x, lam))
    V = decomp$vectors
    L = decomp$values
    return(V %*% diag(sqrt(L)) %*% t(V))
  }
  
  # Check if the constraint is satisfied.
  #check_constraint = function(x, dX){
  #  return(all(x[not_diag(dX)]<0))
  #}  
  
  
  # Inverse of nabla phi
  inv_nabla_phi = function(x0, y,lam){
    g =function(x){
      nabla_phi(x, lam) - y
    }
    # print(nabla_phi(x0,lam)-y)
    x = nleqslv(x0, g)$x
    return(x)
  }
  
  #if(length(x0) != d){
  #  print("Length of x0 is incorrect for data X.")
  #  return()
  #}
  eta = eta0
  
  xs = matrix(numeric(K*d), nrow = d)
  ys = matrix(numeric(K*d), nrow = d)
  all_x = matrix(ncol = 0, nrow = d)
  outer_loop_index = numeric(K)
  
  xs[,1] = x0
  ys[,1] = nabla_phi(x0, lam)
  print(ys[,1])
  outer_loop_index[1] = 1
  
  t1 = Sys.time()
  for(i in 2:K){
    ti = ceiling(multiplier/eta)
    inner_xs = matrix(ncol = ti, nrow = d)
    inner_ys = matrix(ncol = ti, nrow = d) 
    inner_xs[,1] = xs[,i-1]
    inner_ys[,1] = ys[,i-1]
    
    #if(i %% 10 == 0){print(i)}
    if(ti >= 2){
      for (j in 2:ti){
        noi = rnorm(d,0,1)
        inner_ys[,j] = inner_ys[,j-1] - eta*grad(x = inner_xs[,j-1], ...)/scale + 
          sqrt(2*eta/scale)*sHess(inner_xs[,j-1], lam)%*%noi
        # print(grad(x = inner_xs[,j-1], ...))
        inner_xs[,j] = inv_nabla_phi(x0 = inner_xs[,j-1], inner_ys[,j], lam)
      }
    }
    
    xs[,i] = inner_xs[,ti]
    ys[,i] = inner_ys[,ti]
    all_x = cbind(all_x, inner_xs)
    outer_loop_index[i] = outer_loop_index[i-1] + ti - 1
    eta = rho*eta
    
    
    if(message){
      print(paste("iteration:", i ," sample:", paste(xs[,i], collapse = " ")))
    }
  }
  t2 = Sys.time() - t1
  return(list("primal_samples" = xs, "dual_samples" = ys, "time" = t2, "all_x" = all_x, "outer_loop_indices" = outer_loop_index))
}

##
## Integrated function
## All arguments other than ... are necessary for DMLA to run
## If Fn is provided, then we require the first argument of Fn to be the current x
## If grad is provided, we require the function to have the current position x and lam as arguments
## The user must provide one of Fn and grad.
##
sMLA = function(Fn = NULL, grad = NULL, x0, K, eta0, rho, lam, multiplier, A, b, scale, message = F, ...){
  args <- list(x0, K, eta0, rho, lam, multiplier, A, b, scale, message, ...)
  names(args)[1:10] = c("x0","K", "eta0", "rho", "lam", "multiplier", "A", "b", "scale", "message")
  if (!is.null(Fn)) {
    args$Fn = Fn
    if(!"grad_method" %in% names(args) | ! "s0" %in% names(args)){
      print(names(args))
      stop("Missing grad_method or s0")
    }
    if(!is.null(grad)){
      stop("No need to provide both Fn and grad")
    }
    return(do.call(sMLA_new, args))
  }else if (!is.null(grad)) {
    args$grad = grad
    if(! "x" %in% names(formals(grad))){
      stop("The gradient function must the argument: x")
    }
    return(do.call(sMLA_new_grad, args))
  }else{
    stop("Must provide either Fn or grad")
  }
}