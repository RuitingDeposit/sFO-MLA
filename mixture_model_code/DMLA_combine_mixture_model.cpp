// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <chrono>
#include <cmath>
#include <Rcpp.h>
#include <limits>

using namespace Rcpp;

// ============================================================
// Generic functions for the log barrier mirror map.
// ============================================================

// Compute ∇phi(x).
// [[Rcpp::export]]
arma::vec nabla_phi_cpp(
    const arma::vec& x,
    const double lam,
    const arma::mat& A,
    const arma::vec& b
) {
    arma::vec r1 = A * x - b;

    if (arma::any(r1 == 0.0)) {
        stop("nabla_phi is undefined because an entry of A * x - b is zero.");
    }

    return -A.t() * (1.0 / r1) + lam * x;
}

// Compute the Hessian of phi.
// [[Rcpp::export]]
arma::mat Hess_cpp(
    const arma::vec& x,
    const double lam,
    const arma::mat& A,
    const arma::vec& b
) {
    arma::vec r1 = A * x - b;

    if (arma::any(r1 == 0.0)) {
        stop("The Hessian is undefined because an entry of A * x - b is zero.");
    }

    arma::vec diagonal_entries = 1.0 / arma::square(r1);

    // A.t() * diagmat(diagonal_entries) * A, without explicitly
    // constructing the diagonal matrix.
    arma::mat H =
        A.t() * (A.each_col() % diagonal_entries) +
        lam * arma::eye<arma::mat>(x.n_elem, x.n_elem);

    return H;
}

// Compute the symmetric square root of the Hessian.
// [[Rcpp::export]]
arma::mat sHess_cpp(
    const arma::vec& x,
    const double lam,
    const arma::mat& A,
    const arma::vec& b
) {
    arma::mat H = Hess_cpp(x, lam, A, b);

    arma::vec eigenvalues;
    arma::mat eigenvectors;

    bool success = arma::eig_sym(eigenvalues, eigenvectors, H);

    if (!success) {
        stop("The eigendecomposition of the Hessian failed.");
    }

    // Protect against tiny negative eigenvalues caused by rounding.
    const double tolerance = 1e-12;

    if (arma::any(eigenvalues < -tolerance)) {
        stop("The Hessian has a genuinely negative eigenvalue.");
    }

    eigenvalues.transform([](double value) {
        return std::sqrt(std::max(value, 0.0));
    });

    return eigenvectors *
        arma::diagmat(eigenvalues) *
        eigenvectors.t();
}

struct HessInfo {
    arma::mat H;
    arma::mat sqrtH;
    arma::mat invH;
    double logdetH;
};

// Simultaneously return sHess, invHess, and logdetHess
HessInfo HessInfo_impl(
    const arma::vec& x,
    const double lam,
    const arma::mat& A,
    const arma::vec& b
) {
    arma::mat H = Hess_cpp(x, lam, A, b);

    arma::vec eigval;
    arma::mat eigvec;

    if (!arma::eig_sym(eigval, eigvec, H)) {
        Rcpp::stop("The eigendecomposition failed.");
    }

    const double tol = 1e-12;

    if (arma::any(eigval <= tol)) {
        Rcpp::stop(
            "The Hessian is not positive definite."
        );
    }

    arma::vec sqrt_eigval = arma::sqrt(eigval);
    arma::vec inv_eigval = 1.0 / eigval;

    HessInfo result;

    result.H = H;

    result.sqrtH =
        eigvec *
        arma::diagmat(sqrt_eigval) *
        eigvec.t();

    result.invH =
        eigvec *
        arma::diagmat(inv_eigval) *
        eigvec.t();

    result.logdetH = arma::sum(arma::log(eigval));

    return result;
}

// Compute the inverse of ∇phi.
// NEED TO BE MODIFIED HEAVILY
arma::vec inv_nabla_phi_cpp(
    const arma::vec& y,
    double lambda,
    double tol = 1e-10
) {
    const int d = y.n_elem;

    if (d == 0)
        Rcpp::stop("y must be nonempty.");

    if (!std::isfinite(lambda) || lambda <= 0)
        Rcpp::stop("lambda must be finite and positive.");

    if (!std::isfinite(tol) || tol <= 0)
        Rcpp::stop("tol must be finite and positive.");

    for (int i = 0; i < d; ++i) {
        if (!std::isfinite(y[i]))
            Rcpp::stop("All entries of y must be finite.");
    }

    const double c = 2.0 * std::sqrt(lambda);

    // Positive quadratic root, avoiding cancellation.
    auto x_i = [&](double yi, double t) -> double {
        const double a = yi - t;
        const double b = std::hypot(a, c);

        if (a >= 0.0)
            return (0.5 * a + 0.5 * b) / lambda;

        return 1.0 / (0.5 * b - 0.5 * a);
    };

    auto Func = [&](double t) -> double {
        double total = 0.0;
        for (int i = 0; i < d; ++i)
            total += x_i(y[i], t);

        return total + 1.0 / t - 1.0;
    };

    // Find a bracket containing the unique root.
    double lower = 1.0;
    double upper = 2.0;

    while (Func(upper) > 0.0) {
        Rcpp::checkUserInterrupt();

        if (upper > std::numeric_limits<double>::max() / 2.0)
            Rcpp::stop("Could not find a finite upper bracket.");

        upper *= 2.0;
    }

    // Func is strictly decreasing.
    double t = lower + 0.5 * (upper - lower);

    for (int iter = 0; iter < 2048; ++iter) {
        Rcpp::checkUserInterrupt();
        t = lower + 0.5 * (upper - lower);

        // Stop at the requested tolerance or machine precision.
        if (upper - lower <= tol || t == lower || t == upper)
            break;

        const double value = Func(t);

        if (value == 0.0)
            break;

        if (value > 0.0)
            lower = t;
        else
            upper = t;
    }

    arma::vec x(d);
    for (int i = 0; i < d; ++i)
        x[i] = x_i(y[i], t);

    return x;
}

// Determines if the initial point is feasible.
bool is_feasible_grad_cpp(
    const arma::vec& x,
    const arma::mat& A,
    const arma::vec& b
) {
    return arma::all(A * x - b < 0.0);
}


// ============================================================
// Negative log posterior and its gradient
// ============================================================

// [[Rcpp::export]]
arma::vec evaluate_normal_density_cpp(double s, int K, const double std) {
    if (K < 1 || !std::isfinite(s))
        Rcpp::stop("K must be positive and s must be finite.");

    arma::vec dens(K);
    for (int k = 0; k < K; ++k)
        dens[k] = R::dnorm(s, k + 1.0, std, false);
    return dens;
}

// [[Rcpp::export]]
arma::vec grad_est_cpp(
    int m,
    const arma::vec& theta,
    double lam,
    const arma::vec& x,
    const arma::vec& alpha,
    const double std
) {
    const int K = alpha.n_elem;
    const int n = x.n_elem;
    if (K < 2 || theta.n_elem != alpha.n_elem - 1)
        Rcpp::stop("theta must have length length(alpha) - 1.");
    if (m < 1 || m > n)
        Rcpp::stop("m must be between 1 and length(x).");
    if (!theta.is_finite() || arma::any(theta <= 0.0) ||
        arma::sum(theta) >= 1.0)
        Rcpp::stop("theta must be in the simplex interior.");
    if (!alpha.is_finite() || arma::any(alpha <= 0.0))
        Rcpp::stop("alpha must be finite and positive.");
    if (!x.is_finite())
        Rcpp::stop("x must be finite.");

    // Retained for compatibility: the supplied R code does not use lam.
    (void) lam;

    arma::vec weights(K);
    weights.head(K - 1) = theta;
    weights[K - 1] = 1.0 - arma::sum(theta);

    Rcpp::IntegerVector pool = Rcpp::seq_len(n);
    Rcpp::IntegerVector indices = Rcpp::sample(pool, m, false);
    arma::vec grad(K - 1, arma::fill::zeros);
    arma::vec log_dens(K);

    for (int i : indices) {
        Rcpp::checkUserInterrupt();
        for (int k = 0; k < K; ++k)
            log_dens[k] = R::dnorm(x[i - 1], k + 1.0, std, true);

        // A common scale cancels in the density ratio and avoids underflow.
        const double shift = log_dens.max();
        if (!std::isfinite(shift))
            Rcpp::stop("Observation is too extreme to evaluate densities.");
        arma::vec dens = arma::exp(log_dens - shift);
        const double mixture = arma::dot(weights, dens);
        for (int j = 0; j < K - 1; ++j)
            grad[j] -= (dens[j] - dens[K - 1]) / mixture;
    }

    grad *= static_cast<double>(n) / m;
    for (int j = 0; j < K - 1; ++j)
        grad[j] += -(alpha[j] - 1.0) / weights[j]
                   + (alpha[K - 1] - 1.0) / weights[K - 1];
    return grad;
}

// Log posterior up to an additive constant independent of theta.
// [[Rcpp::export]]
double neg_f_cpp(
    const arma::vec& theta,
    const arma::vec& x,
    const arma::vec& alpha,
    const double std
) {
    const int K = alpha.n_elem;
    if (K < 2 || theta.n_elem != alpha.n_elem - 1)
        Rcpp::stop("theta must have length length(alpha) - 1.");
    if (!alpha.is_finite() || arma::any(alpha <= 0.0))
        Rcpp::stop("alpha must be finite and positive.");
    if (!theta.is_finite() || !x.is_finite())
        Rcpp::stop("theta and x must be finite.");

    arma::vec weights(K);
    weights.head(K - 1) = theta;
    weights[K - 1] = 1.0 - arma::sum(theta);
    if (arma::any(weights < 0.0))
        return R_NegInf;
    if (arma::any(weights == 0.0))
        Rcpp::stop("Evaluate on the simplex interior; boundary densities can be singular.");

    arma::vec log_weights = arma::log(weights);
    double result = arma::dot(alpha - 1.0, log_weights);
    arma::vec log_terms(K);
    for (arma::uword i = 0; i < x.n_elem; ++i) {
        Rcpp::checkUserInterrupt();
        for (int k = 0; k < K; ++k)
            log_terms[k] = log_weights[k] +
                R::dnorm(x[i], k + 1.0, std, true);
        const double shift = log_terms.max();
        if (shift == R_NegInf)
            return R_NegInf;
        result += shift + std::log(arma::sum(arma::exp(log_terms - shift)));
    }
    return result;
}
// ============================================================
// Main algorithm when gradient is available. 
// ============================================================

// [[Rcpp::export]]
List DMLA_new_grad_cpp_mix_lglik(
    const arma::vec& theta0,
    const arma::vec& x,
    const arma::vec& alpha,
    const int B,
    const double eta0,
    const double rho,
    const double lam,
    const double multiplier,
    const arma::mat& A,
    const arma::vec& b,
    const double scale,
    const bool message,
    const int m,
    const double std
) {
    RNGScope scope;

    if (B < 1) {stop("B must be at least 1.");}
    if (eta0 <= 0.0) {stop("eta0 must be strictly positive.");}
    if (rho <= 0.0) {stop("rho must be strictly positive.");}
    if (lam <= 0.0) {stop("lam must be strictly positive.");}
    if (multiplier <= 0.0) {stop("multiplier must be strictly positive.");}
    if (scale <= 0.0) {stop("scale must be strictly positive.");}

    const arma::uword d = theta0.n_elem;

    if (A.n_cols != d) {stop("The number of columns of A must equal length(theta0).");}
    if (A.n_rows != b.n_elem) {stop("The number of rows of A must equal length(b).");}
    if (!is_feasible_grad_cpp(theta0, A, b)) {stop("theta0 must satisfy A * theta0 - b < 0.");}

    arma::mat xs(d, B, arma::fill::zeros);
    arma::mat ys(d, B, arma::fill::zeros);
    arma::mat lik(1, B, arma::fill::zeros);
    arma::vec outer_loop_index(B,arma::fill::zeros);

    xs.col(0) = theta0;
    ys.col(0) = nabla_phi_cpp(theta0, lam, A, b);
    lik.col(0) = neg_f_cpp(theta0, x, alpha, std);
    outer_loop_index[0] = 1.0;

    double eta = eta0;

    /*
     * Store blocks and combine them once at the end. This avoids
     * repeatedly reallocating all_x through cbind().
     */
    std::vector<arma::mat> all_x_blocks;
    all_x_blocks.reserve(
        static_cast<std::size_t>(
            std::max(B - 1, 0)
        )
    );

    std::vector<arma::mat> all_lik_blocks;
    all_lik_blocks.reserve(
        static_cast<std::size_t>(
            std::max(B - 1, 0)
        )
    ); //change 

    arma::uword total_all_x_columns = 1;

    // Function do_call("do.call");

    auto start_time =
        std::chrono::steady_clock::now();

    for (int i = 1; i < B; ++i) {
        const int ti =
            static_cast<int>(
                std::ceil(multiplier / eta) + 1
            );

        if (ti < 1) {
            stop("The computed inner-loop length is less than 1.");
        }

        arma::mat inner_xs(d,ti,arma::fill::zeros);
        arma::mat inner_ys(d,ti,arma::fill::zeros);
        arma::mat inner_lik(1,ti,arma::fill::zeros);

        inner_xs.col(0) = xs.col(i - 1);
        inner_ys.col(0) = ys.col(i - 1);
        inner_lik.col(0) = lik.col(i - 1); //change 

        for (int j = 1; j < ti; ++j) {
            arma::vec noise(d);

            for (arma::uword l = 0; l < d; ++l) {
                noise[l] = R::rnorm(0.0, 1.0);
            }

            arma::vec gradient = grad_est_cpp(m,inner_xs.col(j - 1),lam,x,alpha,std); 
            arma::mat sqrt_H = sHess_cpp(inner_xs.col(j - 1),lam,A,b);

            inner_ys.col(j) =
                inner_ys.col(j - 1) -
                eta * gradient / scale +
                std::sqrt(2.0 * eta / scale) *
                sqrt_H * noise;

            inner_xs.col(j) = inv_nabla_phi_cpp(inner_ys.col(j), lam);
            inner_lik.col(j) = neg_f_cpp(inner_xs.col(j), x, alpha, std); 
        }

        xs.col(i) = inner_xs.col(ti - 1);
        ys.col(i) = inner_ys.col(ti - 1);
        lik.col(i) = inner_lik.col(ti - 1);

        all_x_blocks.push_back(inner_xs);
        all_lik_blocks.push_back(inner_lik); 
        total_all_x_columns += ti - 1;

        /*
         * This reproduces the formula in the R code.
         */
        outer_loop_index[i] =
            outer_loop_index[i - 1] +
            ti - 1.0;

        eta *= rho;

        if (message) {
            Rcout
                << "iteration: "
                << i + 1
                << " sample:";

            for (arma::uword l = 0; l < d; ++l) {
                Rcout << " " << xs(l, i);
            }

            Rcout << "\n";
        }
    }

    auto end_time =
        std::chrono::steady_clock::now();

    const double elapsed_seconds =
        std::chrono::duration<double>(
            end_time - start_time
        ).count();

    arma::mat all_x(
        d,
        total_all_x_columns,
        arma::fill::zeros
    );

    arma::mat all_lik(
        1,
        total_all_x_columns,
        arma::fill::zeros
    ); // new

    arma::uword start_column = 0;

    for (const arma::mat& block : all_x_blocks) {
        if (block.n_cols > 0) {
            all_x.cols(
                start_column,
                start_column + block.n_cols - 1
            ) = block;

            start_column += block.n_cols - 1;
        }
    }

    start_column = 0;

    for (const arma::mat& block : all_lik_blocks) {
        if (block.n_cols > 0) {
            all_lik.cols(
                start_column,
                start_column + block.n_cols - 1
            ) = block;

            start_column += block.n_cols - 1;
        }
    } // new 

    return List::create(
        Named("primal_samples") = xs,
        Named("dual_samples") = ys,
        Named("time") = elapsed_seconds,
        Named("all_x") = all_x,
        Named("outer_loop_indices") =
            outer_loop_index,
        Named("all_log_liklihood") = all_lik,
        Named("primal_sample_log_likelihood") = lik
    );
}


// [[Rcpp::export]]
List DMLA_MIX_MH_order1(
    const arma::vec& theta0,
    const arma::vec& x,
    const arma::vec& alpha,
    const int B,
    const double eta0,
    const double lam,
    const arma::mat& A,
    const arma::vec& b,
    const double scale,
    const bool message,
    const int m,
    const double std
) {
    RNGScope scope;

    if (B < 1) {stop("K must be at least 1.");}
    if (eta0 <= 0.0) {stop("eta0 must be strictly positive.");}
    if (lam <= 0.0) {stop("lam must be strictly positive.");}
    if (scale <= 0.0) {stop("scale must be strictly positive.");}

    const arma::uword d = theta0.n_elem;

    if (A.n_cols != d) {stop("The number of columns of A must equal length(x0).");}
    if (A.n_rows != b.n_elem) {stop("The number of rows of A must equal length(b).");}
    if (!is_feasible_grad_cpp(theta0, A, b)) {stop("x0 must satisfy A * theta0 - b < 0.");}

    arma::mat xs(d, B, arma::fill::zeros);
    arma::mat ys(d, B, arma::fill::zeros);
    arma::mat grads(d, B, arma::fill::zeros);
    arma::mat lik(1, B, arma::fill::zeros);

    arma::vec log_accept_vec = arma::zeros<arma::vec>(B);
    Rcpp::LogicalVector accept_vec(B);
    std::vector<HessInfo> all_hess_info(B);

    xs.col(0) = theta0;
    ys.col(0) = nabla_phi_cpp(theta0, lam, A, b);

    grads.col(0) = grad_est_cpp(m,theta0,lam,x,alpha,std); 
    lik.col(0) = neg_f_cpp(theta0, x, alpha, std);
    
    log_accept_vec[0] = 0;
    accept_vec[0] = true;
    all_hess_info[0] = HessInfo_impl(xs.col(0), lam, A, b);

    double eta = eta0;

    auto start_time =
        std::chrono::steady_clock::now();

    for (int i = 1; i < B; ++i) {

        NumericVector noise_r = Rcpp::rnorm(d, 0.0, 1.0);
        arma::vec noise = as<arma::vec>(noise_r);
        
        // Here's what I should modify
        HessInfo hess_old_info = all_hess_info[i-1];
        const arma::mat sqrt_H = hess_old_info.sqrtH;
        const arma::mat inv_hess_old = hess_old_info.invH;
        const double logdet_old = hess_old_info.logdetH;

        ys.col(i) =
            ys.col(i - 1) -
            eta * grads.col(i-1) / scale +
            std::sqrt(2.0 * eta / scale) * sqrt_H * noise;

        xs.col(i) =
            inv_nabla_phi_cpp(ys.col(i), lam);
        
        
        lik.col(i) = neg_f_cpp(xs.col(i), x, alpha, std); 

        HessInfo hess_new_info = HessInfo_impl(xs.col(i), lam, A, b);
        all_hess_info[i] = hess_new_info;
        const arma::mat inv_hess_new = hess_new_info.invH;
        const double logdet_new = hess_new_info.logdetH;

        // The MH step 
        /////
        /////
        // 
            grads.col(i) = grad_est_cpp(m,xs.col(i),lam,x,alpha,std);

            const arma::vec residual_old =
                ys.col(i) - ys.col(i-1) + eta * grads.col(i-1) / scale;
            const arma::vec residual_new =
                ys.col(i-1) - ys.col(i) + eta * grads.col(i) / scale;
            
            const double log_frac_1 =
                arma::dot(
                    residual_old,
                    inv_hess_old * residual_old
                ) / (4.0 * eta / scale)
                -
                arma::dot(
                    residual_new,
                    inv_hess_new * residual_new
                ) / (4.0 * eta / scale);
            
            const double log_frac_2 =
                1.5 * (
                    logdet_old -
                    logdet_new
                );
            const double log_proposal_diff =
                log_frac_1 +
                log_frac_2;

            
            // Compute log posterior difference
            // Note that Fn is minus log posterior
            double log_posterior_diff = neg_f_cpp(xs.col(i), x, alpha, std) - neg_f_cpp(xs.col(i-1),x, alpha, std);
            const double log_alpha = log_proposal_diff + log_posterior_diff;
            bool accept = std::log(R::runif(0.0, 1.0)) <= log_alpha;
            
            log_accept_vec[i] = log_alpha;
            accept_vec[i] = accept;

        /////
        /////
        // End MH step
            
        if(!accept){
            xs.col(i) = xs.col(i-1);
            ys.col(i) = ys.col(i-1);
            all_hess_info[i] = hess_old_info;
            grads.col(i) = grads.col(i-1);
            lik.col(i) = lik.col(i-1);
        }

        if (message && (i+1)%100 == 0) {
            Rcout << "iteration: " << i + 1 << " sample:";

            for (arma::uword k = 0; k < d; ++k) {
                Rcout << " " << xs(k, i);
            }

            Rcout << "\n";
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return List::create(
        Named("primal_samples") = xs,
        Named("dual_samples") = ys,
        Named("time") = elapsed_seconds,
        Named("log_acceptance_ratio") = log_accept_vec,
        Named("gradients") = grads,
        Named("acceptance") = accept_vec,
        Named("log_likelihood") = lik
    );     
}