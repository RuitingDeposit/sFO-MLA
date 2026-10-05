// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#include <chrono>
#include <cmath>

using namespace Rcpp;

// From gradient_est
// From PGG
// ------------------------------------------------------------
// Convert vech(theta) into a symmetric matrix.
//
// Assumption:
// theta_vec contains the lower triangle column by column:
//
// theta11,
// theta21, theta31, ...,
// theta22,
// theta32, ...,
// ...
// ------------------------------------------------------------

arma::mat invvech_cpp(const arma::vec& theta_vec) {

    const arma::uword m = theta_vec.n_elem;

    // Solve d(d+1)/2 = m
    const double d_real =
        (-1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(m))) / 2.0;

    const arma::uword d =
        static_cast<arma::uword>(std::round(d_real));

    if (d * (d + 1) / 2 != m) {
        stop("length(theta_vec) must equal d(d+1)/2 for some integer d.");
    }

    arma::mat theta(d, d, arma::fill::zeros);

    arma::uword index = 0;

    for (arma::uword col = 0; col < d; ++col) {
        for (arma::uword row = col; row < d; ++row) {

            theta(row, col) = theta_vec[index];
            theta(col, row) = theta_vec[index];

            ++index;
        }
    }

    return theta;
}

// ------------------------------------------------------------
// vech: upper triangle, row by row
//
// For a symmetric 3 x 3 matrix S, this returns
// S(1,1), S(1,2), S(1,3), S(2,2), S(2,3), S(3,3)
//
// This matches the ordering used by your R loops and by the
// previously defined invvech_cpp().
// ------------------------------------------------------------

arma::vec vech_cpp_internal(const arma::mat& S) {
    if (S.n_rows != S.n_cols) {
        Rcpp::stop("S must be a square matrix.");
    }

    const arma::uword d = S.n_rows;
    const arma::uword p = d * (d + 1) / 2;

    arma::vec result(p);
    arma::uword count = 0;

    for (arma::uword i = 0; i < d; ++i) {
        for (arma::uword j = i; j < d; ++j) {
            result[count++] = S(i, j);
        }
    }

    return result;
}

// [[Rcpp::export]]
arma::mat PGMsim_cpp(
    const int n,
    const arma::mat& theta,
    const int max_iter
) {
    RNGScope scope;

    if (n <= 0) {
        stop("n must be positive.");
    }

    if (max_iter < 0) {
        stop("max_iter must be nonnegative.");
    }

    if (theta.n_rows != theta.n_cols) {
        stop("theta must be a square matrix.");
    }

    const arma::uword d = theta.n_cols;

    arma::mat X(n, d);

    // Initial values:
    // X[, j] ~ Poisson(exp(theta[j,j]))
    for (arma::uword j = 0; j < d; ++j) {
        const double initial_rate = std::exp(theta(j, j));

        if (!std::isfinite(initial_rate)) {
            stop("A diagonal entry of theta produces a nonfinite Poisson rate.");
        }

        for (int i = 0; i < n; ++i) {
            X(i, j) = R::rpois(initial_rate);
        }
    }

    for (int k = 0; k < max_iter; ++k) {
        for (arma::uword j = 0; j < d; ++j) {

            /*
             * Since theta is symmetric,
             *
             * theta[j,j] + 2 * X[, -j] %*% theta[-j,j]
             *
             * can be computed as
             *
             * theta[j,j] + 2 * sum_{l != j} X[,l] theta[l,j].
             */

            arma::vec linear_predictor(n);
            linear_predictor.fill(theta(j, j));

            for (arma::uword l = 0; l < d; ++l) {
                if (l != j) {
                    linear_predictor +=
                        2.0 * theta(l, j) * X.col(l);
                }
            }

            arma::vec rate_j = arma::exp(linear_predictor);

            for (int i = 0; i < n; ++i) {
                if (!std::isfinite(rate_j[i]) || rate_j[i] < 0.0) {
                    stop(
                        "A nonfinite or invalid Poisson rate was generated "
                        "at iteration %d, column %d, row %d.",
                        k + 1,
                        j + 1,
                        i + 1
                    );
                }

                X(i, j) = R::rpois(rate_j[i]);
            }
        }
    }

    return X;
}

// Compute the Hessian for PGG specifically
arma::vec Hess_PGG_cpp(
    const arma::vec& x,
    const int dX,
    const double lam
) {
    if (dX < 1) {
        stop("dX must be positive.");
    }

    const arma::uword d =
        static_cast<arma::uword>(dX * (dX + 1) / 2);

    if (x.n_elem != d) {
        stop("length(x) must equal dX * (dX + 1) / 2.");
    }

    arma::vec result(d);
    result.fill(lam);

    arma::uword position = 0;

    /*
     * Reproduces not_diag(dX):
     *
     * dX = 3 gives
     * 0, 1, 1, 0, 1, 0
     */
    for (int block_size = dX; block_size >= 1; --block_size) {

        // First element of each block corresponds to a diagonal entry.
        ++position;

        // Remaining elements correspond to off-diagonal entries.
        for (int j = 1; j < block_size; ++j) {
            if (x[position] == 0.0) {
                stop("An off-diagonal entry of x is zero.");
            }

            result[position] +=
                1.0 / (x[position] * x[position]);

            ++position;
        }
    }

    return result;
}

Rcpp::LogicalVector not_diag_cpp(int dX) {

    if (dX < 1) {
        Rcpp::stop("dX must be positive.");
    }

    if (dX == 1) {
        return Rcpp::LogicalVector::create(false);
    }

    const int len = dX * (dX + 1) / 2;
    Rcpp::LogicalVector v(len);

    int pos = 0;

    for (int i = dX - 1; i >= 0; --i) {

        v[pos++] = false;

        for (int j = 0; j < i; ++j) {
            v[pos++] = true;
        }
    }

    return v;
}

arma::vec logz_gradient_estimate_cpp(
    const int n,
    const arma::mat& theta,
    const int max_iter,
    const int B
) {
    Rcpp::RNGScope scope;

    if (n <= 0) {
        Rcpp::stop("n must be positive.");
    }

    if (max_iter < 0) {
        Rcpp::stop("max_iter must be nonnegative.");
    }

    if (theta.n_rows != theta.n_cols) {
        Rcpp::stop("theta must be square.");
    }

    /*
     * B is interpreted exactly like the R argument:
     * B = 1 means use rows 1,...,n.
     */
    if (B < 1 || B > n) {
        Rcpp::stop("B must satisfy 1 <= B <= n.");
    }

    const arma::uword d = theta.n_rows;
    const arma::uword p = d * (d + 1) / 2;

    arma::vec gradient(p, arma::fill::zeros);

    /*
     * PGMsim_cpp must have been defined earlier in the file.
     */
    arma::mat X = PGMsim_cpp(n, theta, max_iter);

    // Convert the R-style index B to a zero-based C++ index.
    const arma::uword first_row =
        static_cast<arma::uword>(B - 1);

    const arma::uword number_retained =
        static_cast<arma::uword>(n - B + 1);

    arma::uword count = 0;

    for (arma::uword i = 0; i < d; ++i) {
        for (arma::uword j = i; j < d; ++j) {

            if (i == j) {
                double total = 0.0;

                for (arma::uword row = first_row;
                     row < static_cast<arma::uword>(n);
                     ++row) {
                    total += X(row, i);
                }

                gradient[count] =
                    total / static_cast<double>(number_retained);

            } else {
                double total = 0.0;

                for (arma::uword row = first_row;
                     row < static_cast<arma::uword>(n);
                     ++row) {
                    total += X(row, i) * X(row, j);
                }

                gradient[count] =
                    2.0 * total /
                    static_cast<double>(number_retained);
            }

            ++count;
        }
    }

    return gradient;
}

// ------------------------------------------------------------
// Accurate estimate of the transformed gradient
// -
// [[Rcpp::export]]
arma::vec grad_est_acc_cpp(
    const arma::vec& x,
    const double lam,
    const int n,
    const arma::mat& S,
    const int dX,
    const int N_samp,
    const double prior = 10.0,
    const int max_iter = 500
) {
    Rcpp::RNGScope scope;

    if (dX < 1) {
        Rcpp::stop("dX must be positive.");
    }

    if (n <= 0) {
        Rcpp::stop("n must be positive.");
    }

    if (N_samp < 2) {
        Rcpp::stop("N_samp must be at least 2.");
    }

    if (prior <= 0.0) {
        Rcpp::stop("prior must be positive.");
    }

    if (max_iter < 0) {
        Rcpp::stop("max_iter must be nonnegative.");
    }

    if (S.n_rows != static_cast<arma::uword>(dX) ||
        S.n_cols != static_cast<arma::uword>(dX)) {
        Rcpp::stop("S must be a dX by dX matrix.");
    }

    const arma::uword p =
        static_cast<arma::uword>(dX * (dX + 1) / 2);

    if (x.n_elem != p) {
        Rcpp::stop(
            "length(x) must equal dX * (dX + 1) / 2."
        );
    }

    arma::mat curr_theta = invvech_cpp(x);
    arma::vec vech_S = vech_cpp_internal(S);

    /*
     * Reproduce:
     *
     * ifelse(not_diag(dX), 2, 1)
     */
    arma::vec multiplicity(p, arma::fill::ones);

    arma::uword count = 0;

    for (int i = 0; i < dX; ++i) {
        for (int j = i; j < dX; ++j) {
            if (i != j) {
                multiplicity[count] = 2.0;
            }

            ++count;
        }
    }

    arma::vec estimate =
        -vech_S % multiplicity +
        x / (prior * prior);

    /*
     * Your R code uses B = N_samp / 2.
     *
     * Integer division here gives floor(N_samp / 2). For exact
     * agreement without ambiguity, use an even N_samp.
     */
    const int B = N_samp / 2;

    arma::vec gradient_z =
        static_cast<double>(n) *
        logz_gradient_estimate_cpp(
            N_samp,
            curr_theta,
            max_iter,
            B
        );

    estimate = (estimate + gradient_z);

    return estimate;
}


// ------------------------------------------------------------
// Unbiased estimate of z(theta)/z(phi)
// ------------------------------------------------------------

double ztheta_cpp(
    const arma::mat& theta,
    const int N
) {
    if (N <= 0) {
        stop("N must be positive.");
    }

    if (theta.n_rows != theta.n_cols) {
        stop("theta must be square.");
    }

    const arma::uword d = theta.n_rows;
    const arma::vec phi = theta.diag();

    // Remove the diagonal from theta.
    arma::mat off_diagonal_theta = theta;
    off_diagonal_theta.diag().zeros();

    double sum_weights = 0.0;

    /*
     * We do not need to store the entire N by d matrix Y.
     * Generate one importance sample at a time.
     */
    for (int i = 0; i < N; ++i) {

        arma::vec y(d);

        for (arma::uword j = 0; j < d; ++j) {
            y[j] = R::rpois(std::exp(phi[j]));
        }

        const double quadratic_form =
            arma::as_scalar(y.t() * off_diagonal_theta * y);

        sum_weights += std::exp(quadratic_form);
    }

    return sum_weights / static_cast<double>(N);
}


// ------------------------------------------------------------
// Unbiased estimate of log z(theta)/z(phi)
// ------------------------------------------------------------

double log_ztheta_cpp(
    const arma::mat& theta,
    const int N,
    const double p0
) {
    if (p0 <= 0.0 || p0 >= 1.0) {
        stop("p0 must lie strictly between 0 and 1.");
    }

    if (N <= 0) {
        stop("N must be positive.");
    }

    /*
     * R::rgeom(p0) has support {0,1,2,...}, matching rgeom(1,p0).
     */
    const int R_value =
        static_cast<int>(R::rgeom(p0)) + 1;

    const double a = ztheta_cpp(theta, N);

    if (!std::isfinite(a) || a <= 0.0) {
        stop("ztheta produced a nonpositive or nonfinite value.");
    }

    double estimate = std::log(a);

    for (int k = 1; k <= R_value; ++k) {

        double product_term = 1.0;

        for (int l = 0; l < k; ++l) {

            const double T_value = ztheta_cpp(theta, N);

            product_term *= (T_value - a) / a;
        }

        const double sign =
            (k % 2 == 1) ? 1.0 : -1.0;

        const double denominator =
            static_cast<double>(k) *
            std::pow(1.0 - p0, k - 1);

        estimate += sign * product_term / denominator;
    }

    return estimate;
}


// ------------------------------------------------------------
// Log-density of N(0,sd^2)
// ------------------------------------------------------------

double log_normal_density_zero_mean(
    const double x,
    const double sd
) {
    return R::dnorm4(x, 0.0, sd, true);
}


// ------------------------------------------------------------
// Log-density of N(0,sd^2), truncated to (-Inf,0]
// ------------------------------------------------------------

double log_upper_zero_truncated_normal(
    const double x,
    const double sd
) {
    if (x > 0.0) {
        return R_NegInf;
    }

    // P[N(0,sd^2) <= 0] = 1/2
    return R::dnorm4(x, 0.0, sd, true) + std::log(2.0);
}

// ------------------
// Compute log prior (d not declared here)
// ------------------

double log_prior(
    const arma::vec& theta_vec,
    const double prior_sd = 10.0
){
    double lgprior = 0.0;
    arma::mat theta = invvech_cpp(theta_vec);
    int d = theta.n_rows;

    for (arma::uword i = 0; i < d; ++i) {
        lgprior += log_normal_density_zero_mean(
            theta(i, i),
            prior_sd
        );
    }

    for (arma::uword col = 0; col < d; ++col) {
        for (arma::uword row = col + 1; row < d; ++row) {
            const double log_density =
                log_upper_zero_truncated_normal(
                    theta(row, col),
                    prior_sd
                );

            if (!std::isfinite(log_density)) {
                return R_PosInf;
            }

            lgprior += log_density;
        }
    }

    return lgprior;
}

// ------------------
// Compute log E
// ------------------

double log_E(
    const arma::vec& theta_vec,
    const arma::mat& X
) {
    arma::mat theta = invvech_cpp(theta_vec);
    if (theta.n_rows != X.n_cols) {
        stop("theta_vec and X imply different dimensions.");
    }

    arma::mat S = X.t() * X;
    S.diag() = arma::sum(X, 0).t();

    return arma::trace(S * theta);
}


// ------------------------------------------------------------
// Unbiased estimator of negative log-posterior
// ------------------------------------------------------------

// [[Rcpp::export]]
double unbiased_log_posterior_cpp(
    const arma::vec& theta_vec,
    const arma::mat& X,
    const int N = 1000,
    const int M = 10,
    const double p0 = 0.99,
    const double prior_sd = 10.0
) {
    /*
     * M is retained to match the R function interface, but it is not
     * used in the original R code.
     */
    (void) M;

    if (N <= 0) {
        stop("N must be positive.");
    }

    if (prior_sd <= 0.0) {
        stop("prior_sd must be positive.");
    }

    const arma::uword n = X.n_rows;
    const arma::uword d = X.n_cols;

    arma::mat theta = invvech_cpp(theta_vec);

    if (theta.n_rows != d) {
        stop("theta_vec and X imply different dimensions.");
    }

    // S = X'X
    arma::mat S = X.t() * X;

    // Replace diagonal with column sums of X.
    S.diag() = arma::sum(X, 0).t();

    /*
     * sum(diag(S %*% theta)) = trace(S theta)
     */
    const double t1 = arma::trace(S * theta);

    const double estimated_log_ratio =
        log_ztheta_cpp(theta, N, p0);

    const double log_z_phi =
        arma::sum(arma::exp(theta.diag()));

    const double t2 =
        static_cast<double>(n) *
        (estimated_log_ratio + log_z_phi);

    double t3 = 0.0;

    // Normal prior on diagonal entries.
    for (arma::uword i = 0; i < d; ++i) {
        t3 += log_normal_density_zero_mean(
            theta(i, i),
            prior_sd
        );
    }

    // Truncated-normal prior on strict lower triangle.
    for (arma::uword col = 0; col < d; ++col) {
        for (arma::uword row = col + 1; row < d; ++row) {

            const double log_density =
                log_upper_zero_truncated_normal(
                    theta(row, col),
                    prior_sd
                );

            if (!std::isfinite(log_density)) {
                return R_PosInf;
            }

            t3 += log_density;
        }
    }

    return -t1 + t2 - t3;
}


// ------------------------------------------------------------
// Biased estimator of negative log-posterior
// ------------------------------------------------------------

// [[Rcpp::export]]
double biased_log_posterior_cpp(
    const arma::vec& theta_vec,
    const arma::mat& X,
    const int N = 1000,
    const double p0 = 0.7,
    const double prior_sd = 10.0
) {
    /*
     * p0 is retained only to match the original R interface.
     * It is unused in biased_log_posterior().
     */
    (void) p0;

    if (N <= 0) {
        stop("N must be positive.");
    }

    if (prior_sd <= 0.0) {
        stop("prior_sd must be positive.");
    }

    const arma::uword n = X.n_rows;
    const arma::uword d = X.n_cols;

    arma::mat theta = invvech_cpp(theta_vec);

    if (theta.n_rows != d) {
        stop("theta_vec and X imply different dimensions.");
    }

    arma::mat S = X.t() * X;
    S.diag() = arma::sum(X, 0).t();

    const double t1 = arma::trace(S * theta);

    const double z_estimate = ztheta_cpp(theta, N);

    if (!std::isfinite(z_estimate) || z_estimate <= 0.0) {
        stop("ztheta produced a nonpositive or nonfinite value.");
    }

    const double t2 =
        static_cast<double>(n) *
        (
            std::log(z_estimate) +
            arma::sum(arma::exp(theta.diag()))
        );

    double t3 = 0.0;

    for (arma::uword i = 0; i < d; ++i) {
        t3 += log_normal_density_zero_mean(
            theta(i, i),
            prior_sd
        );
    }

    for (arma::uword col = 0; col < d; ++col) {
        for (arma::uword row = col + 1; row < d; ++row) {

            const double log_density =
                log_upper_zero_truncated_normal(
                    theta(row, col),
                    prior_sd
                );

            if (!std::isfinite(log_density)) {
                return R_PosInf;
            }

            t3 += log_density;
        }
    }

    return -t1 + t2 - t3;
}

//===========
// From DMLA
//
//
//
//
// Evaluate Fn(x, ...) using R's do.call().
double evaluate_Fn(
    const Function& Fn,
    const arma::vec& x,
    const List& dots,
    const Function& do_call
) {
    List args(dots.size() + 1);

    // The first argument to Fn is x.
    args[0] = wrap(x);

    // Add the arguments from ...
    CharacterVector dot_names = dots.names();

    for (R_xlen_t i = 0; i < dots.size(); ++i) {
        args[i + 1] = dots[i];
    }

    // Preserve names of the ... arguments.
    if (dot_names.size() == dots.size()) {
        CharacterVector arg_names(dots.size() + 1);
        arg_names[0] = "";

        for (R_xlen_t i = 0; i < dots.size(); ++i) {
            arg_names[i + 1] = dot_names[i];
        }

        args.attr("names") = arg_names;
    }

    return as<double>(do_call(Fn, args));
}

// Evaluate grad(x, lam, ...) using R's do.call().
arma::vec evaluate_grad_R(
    const Rcpp::Function& grad,
    const arma::vec& x,
    const double lam,
    const Rcpp::List& dots,
    const Rcpp::Function& do_call
) {
    Rcpp::List args(dots.size() + 2);

    args[0] = Rcpp::wrap(x);
    args[1] = lam;

    Rcpp::CharacterVector arg_names(dots.size() + 2);
    arg_names[0] = "x";
    arg_names[1] = "lam";

    Rcpp::CharacterVector dot_names = dots.names();

    for (R_xlen_t i = 0; i < dots.size(); ++i) {
        args[i + 2] = dots[i];

        if (dot_names.size() == dots.size()) {
            arg_names[i + 2] = dot_names[i];
        } else {
            arg_names[i + 2] = "";
        }
    }

    args.attr("names") = arg_names;

    SEXP result = do_call(grad, args);

    return Rcpp::as<arma::vec>(result);
}

// Compute ∇phi(x).
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
// Need to be modified for nleqslv
arma::vec inv_nabla_phi_cpp(
    const arma::vec& y,
    const double lam,
    const arma::uvec& non_diag
) {
    if (lam <= 0.0) {
        stop("lam must be strictly positive.");
    }

    arma::vec z(y.n_elem);

    for (arma::uword i = 0; i < y.n_elem; ++i) {
        if (non_diag[i] != 0u) {
            z[i] =
                (y[i] - std::sqrt(y[i] * y[i] + 4.0 * lam)) /
                (2.0 * lam);
        } else {
            z[i] = y[i] / lam;
        }
    }

    return z;
}

// Determines if the initial point is feasible.
bool is_feasible_grad_cpp(
    const arma::vec& x,
    const arma::mat& A,
    const arma::vec& b
) {
    return arma::all(A * x - b < 0.0);
}

// Centered finite-difference gradient estimate.
arma::vec grad_est_centered_cpp(
    const Function& Fn,
    const arma::vec& x,
    const double eps,
    const double lam,
    const arma::mat& A,
    const arma::vec& b,
    const arma::uvec& non_diag,
    const List& dots,
    const Function& do_call
) {
    arma::uword d = x.n_elem;
    arma::vec estimate(d);

    arma::vec dual_x = nabla_phi_cpp(x, lam, A, b);

    for (arma::uword i = 0; i < d; ++i) {
        arma::vec y_minus = dual_x;
        arma::vec y_plus  = dual_x;

        y_minus[i] -= eps;
        y_plus[i]  += eps;

        arma::vec x1 = inv_nabla_phi_cpp(y_minus, lam, non_diag);
        arma::vec x2 = inv_nabla_phi_cpp(y_plus, lam, non_diag);

        double F1 = evaluate_Fn(Fn, x1, dots, do_call);
        double F2 = evaluate_Fn(Fn, x2, dots, do_call);

        estimate[i] = (F2 - F1) / (2.0 * eps);
    }

    return Hess_cpp(x,lam,A,b)*estimate;
}

// Forward finite-difference gradient estimate.
arma::vec grad_est_forward_cpp(
    const Function& Fn,
    const arma::vec& x,
    const double eps,
    const double lam,
    const arma::mat& A,
    const arma::vec& b,
    const arma::uvec& non_diag,
    const List& dots,
    const Function& do_call
) {
    arma::uword d = x.n_elem;
    arma::vec estimate(d);

    double F0 = evaluate_Fn(Fn, x, dots, do_call);
    arma::vec dual_x = nabla_phi_cpp(x, lam, A, b);

    for (arma::uword i = 0; i < d; ++i) {
        arma::vec y_plus = dual_x;
        y_plus[i] += eps;

        arma::vec x1 = inv_nabla_phi_cpp(y_plus, lam, non_diag);
        double F1 = evaluate_Fn(Fn, x1, dots, do_call);

        estimate[i] = (F1 - F0) / eps;
    }

    return  Hess_cpp(x,lam,A,b)*estimate;
}

// ============================================================
// Main algorithm when gradient is not available. 
// ============================================================

// [[Rcpp::export]]
List DMLA_new_cpp_impl_lglik(
    Function Fn,
    const arma::vec& x0,
    const int K,
    const arma::mat& dat, 
    const double eta0,
    const double rho,
    const double lam,
    const std::string grad_method,
    const double s0,
    const double multiplier,
    const arma::mat& A,
    const arma::vec& b,
    const double scale,
    const bool message,
    const LogicalVector non_diag_r,
    const bool MH,
    const List dots = List::create()
) {
    RNGScope scope;

    // Logistics to keep the function robust to user input.
    //////
    //////
    if (K < 1) {
        stop("K must be at least 1.");
    }
    if (eta0 <= 0.0) {
        stop("eta0 must be strictly positive.");
    }
    if (rho <= 0.0) {
        stop("rho must be strictly positive.");
    }
    if (lam <= 0.0) {
        stop("lam must be strictly positive.");
    }
    if (scale <= 0.0) {
        stop("scale must be strictly positive.");
    }
    if (multiplier <= 0.0) {
        stop("multiplier must be strictly positive.");
    }

    arma::uword d = x0.n_elem;

    if (A.n_cols != d) {
        stop("The number of columns of A must equal length(x0).");
    }
    if (A.n_rows != b.n_elem) {
        stop("The number of rows of A must equal length(b).");
    }
    if (static_cast<arma::uword>(non_diag_r.size()) != d) {
        stop("length(non_diag) must equal length(x0).");
    }

    arma::uvec non_diag(d);

    for (arma::uword i = 0; i < d; ++i) {
        if (LogicalVector::is_na(non_diag_r[i])) {
            stop("non_diag cannot contain NA.");
        }

        non_diag[i] = non_diag_r[i] ? 1u : 0u;
    }
    if (!is_feasible_grad_cpp(x0, A, b)) {
        stop("x0 must satisfy A * x0 - b < 0.");
    }
    //////
    //////
    // End logistics 

    bool use_centered;

    if (grad_method == "center" || grad_method == "centered") {
        use_centered = true;
    } else if (grad_method == "forward") {
        use_centered = false;
    } else {
        stop("grad_method must be 'center', 'centered', or 'forward'.");
    }

    double power = use_centered ? 1.0 / 6.0 : 1.0 / 4.0;
    double eta = eta0;

    arma::mat xs(d, K, arma::fill::zeros);
    arma::mat ys(d, K, arma::fill::zeros);
    arma::mat lik(1, K, arma::fill::zeros);
    xs.col(0) = x0;
    ys.col(0) = nabla_phi_cpp(x0, lam, A, b);
    lik.col(0) = log_E(x0, dat) - dat.n_rows*(log(ztheta_cpp(invvech_cpp(x0), 5000)) + arma::sum(arma::exp(invvech_cpp(x0).diag()))) + log_prior(x0); 

    arma::vec outer_loop_index(K, arma::fill::zeros);
    outer_loop_index[0] = 1.0;
    /*
    * Unlike the original repeated cbind(), store each inner trajectory in
    * a field and join them once at the end. This avoids repeated memory
    * allocation.
    */
    std::vector<arma::mat> all_x_blocks;
    all_x_blocks.reserve(std::max(K - 1, 0));
    
    std::vector<arma::mat> all_lik_blocks;
    all_lik_blocks.reserve(
        static_cast<std::size_t>(
            std::max(K - 1, 0)
        )
    );

    arma::uword total_all_x_columns = 1;

    Function do_call("do.call");
    auto start_time = std::chrono::steady_clock::now();

    for (int i = 1; i < K; ++i) {
        int ti = static_cast<int>(std::ceil(multiplier / eta))+1;

        if (ti < 1) {
            stop("The computed number of inner iterations is less than 1.");
        }

        arma::mat inner_xs(d, ti, arma::fill::zeros);
        arma::mat inner_ys(d, ti, arma::fill::zeros);
        arma::mat inner_lik(1, ti, arma::fill::zeros);

        inner_xs.col(0) = xs.col(i - 1);
        inner_ys.col(0) = ys.col(i - 1);
        inner_lik.col(0) = lik.col(i - 1); 

        double eps = s0 * std::pow(eta, power);

        if (!std::isfinite(eps) || eps <= 0.0) {
            stop("The finite-difference step eps is not positive and finite.");
        }

        for (int j = 1; j < ti; ++j) {
            arma::vec gradient;

            if (use_centered) {
                gradient = grad_est_centered_cpp(
                    Fn,
                    inner_xs.col(j - 1),
                    eps,
                    lam,
                    A,
                    b,
                    non_diag,
                    dots,
                    do_call
                );
            } else {
                gradient = grad_est_forward_cpp(
                    Fn,
                    inner_xs.col(j - 1),
                    eps,
                    lam,
                    A,
                    b,
                    non_diag,
                    dots,
                    do_call
                );
            }

            NumericVector noise_r = Rcpp::rnorm(d, 0.0, 1.0);
            arma::vec noise = as<arma::vec>(noise_r);

            arma::mat sqrt_H = sHess_cpp(
                inner_xs.col(j - 1),
                lam,
                A,
                b
            );

            inner_ys.col(j) =
                inner_ys.col(j - 1) -
                eta * gradient / scale +
                std::sqrt(2.0 * eta / scale) * sqrt_H * noise;

            inner_xs.col(j) =
                inv_nabla_phi_cpp(inner_ys.col(j), lam, non_diag);

            inner_lik.col(j) = log_E(inner_xs.col(j), dat) - dat.n_rows*(log(ztheta_cpp(invvech_cpp(inner_xs.col(j)), 5000))+arma::sum(arma::exp(invvech_cpp(inner_xs.col(j)).diag()))) + log_prior(inner_xs.col(j)); 
        }

        xs.col(i) = inner_xs.col(ti - 1);
        ys.col(i) = inner_ys.col(ti - 1);
        lik.col(i) = inner_lik.col(ti - 1);

        all_x_blocks.push_back(inner_xs);
        all_lik_blocks.push_back(inner_lik); 

        total_all_x_columns += ti - 1;

        // This reproduces the indexing formula in your R code.
        outer_loop_index[i] =
            outer_loop_index[i - 1] + ti - 1.0;

        eta *= rho;

        if (message) {
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

    arma::mat all_x(d, total_all_x_columns, arma::fill::zeros);
    arma::mat all_lik(1, total_all_x_columns, arma::fill::zeros); // new

    arma::uword current_column = 0;

    for (const arma::mat& block : all_x_blocks) {
        if (block.n_cols > 0) {
            all_x.cols(
                current_column,
                current_column + block.n_cols - 1
            ) = block;

            current_column += block.n_cols - 1;
        }
    }

    current_column = 0;

    for (const arma::mat& block : all_lik_blocks) {
        if (block.n_cols > 0) {
            all_lik.cols(
                current_column,
                current_column + block.n_cols - 1
            ) = block;

            current_column += block.n_cols - 1;
        }
    } // new 

    return List::create(
        Named("primal_samples") = xs,
        Named("dual_samples") = ys,
        Named("time") = elapsed_seconds,
        Named("all_x") = all_x,
        Named("outer_loop_indices") = outer_loop_index,
        Named("all_log_liklihood") = all_lik,
        Named("primal_sample_log_likelihood") = lik
    );

}

// ============================================================
// Main algorithm when gradient is available. 
// ============================================================

// [[Rcpp::export]]
List DMLA_new_grad_cpp_impl_lglik(
    Function grad,
    const arma::vec& x0,
    const arma::mat& X,
    const int K,
    const double eta0,
    const double rho,
    const double lam,
    const double multiplier,
    const arma::mat& A,
    const arma::vec& b,
    const double scale,
    const bool message,
    const LogicalVector non_diag_r,
    const List& dots
) {
    RNGScope scope;

    if (K < 1) {
        stop("K must be at least 1.");
    }

    if (eta0 <= 0.0) {
        stop("eta0 must be strictly positive.");
    }

    if (rho <= 0.0) {
        stop("rho must be strictly positive.");
    }

    if (lam <= 0.0) {
        stop("lam must be strictly positive.");
    }

    if (multiplier <= 0.0) {
        stop("multiplier must be strictly positive.");
    }

    if (scale <= 0.0) {
        stop("scale must be strictly positive.");
    }

    const arma::uword d = x0.n_elem;

    if (A.n_cols != d) {
        stop(
            "The number of columns of A must equal length(x0)."
        );
    }

    if (A.n_rows != b.n_elem) {
        stop(
            "The number of rows of A must equal length(b)."
        );
    }

    if (static_cast<arma::uword>(non_diag_r.size()) != d) {
        stop("length(non_diag) must equal length(x0).");
    }

    arma::uvec non_diag(d);

    for (arma::uword i = 0; i < d; ++i) {
        if (LogicalVector::is_na(non_diag_r[i])) {
            stop("non_diag cannot contain NA.");
        }

        non_diag[i] = non_diag_r[i] ? 1u : 0u;
    }

    if (!is_feasible_grad_cpp(x0, A, b)) {
        stop("x0 must satisfy A * x0 - b < 0.");
    }

    arma::mat xs(d, K, arma::fill::zeros);
    arma::mat ys(d, K, arma::fill::zeros);
    arma::mat lik(1, K, arma::fill::zeros);

    arma::vec outer_loop_index(
        K,
        arma::fill::zeros
    );

    xs.col(0) = x0;
    ys.col(0) =
        nabla_phi_cpp(x0, lam, A, b);
    lik.col(0) = log_E(x0, X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(x0), 5000)) + arma::sum(arma::exp(invvech_cpp(x0).diag()))) + log_prior(x0);
    
    outer_loop_index[0] = 1.0;

    double eta = eta0;

    /*
     * Store blocks and combine them once at the end. This avoids
     * repeatedly reallocating all_x through cbind().
     */
    std::vector<arma::mat> all_x_blocks;
    all_x_blocks.reserve(
        static_cast<std::size_t>(
            std::max(K - 1, 0)
        )
    );

    std::vector<arma::mat> all_lik_blocks;
    all_lik_blocks.reserve(
        static_cast<std::size_t>(
            std::max(K - 1, 0)
        )
    ); //change 

    arma::uword total_all_x_columns = 1;

    Function do_call("do.call");

    auto start_time =
        std::chrono::steady_clock::now();

    for (int i = 1; i < K; ++i) {
        const int ti =
            static_cast<int>(
                std::ceil(multiplier / eta) + 1
            );

        if (ti < 1) {
            stop("The computed inner-loop length is less than 1.");
        }

        arma::mat inner_xs(
            d,
            ti,
            arma::fill::zeros
        );

        arma::mat inner_ys(
            d,
            ti,
            arma::fill::zeros
        );

        arma::mat inner_lik(
            1,
            ti,
            arma::fill::zeros
        );

        inner_xs.col(0) = xs.col(i - 1);
        inner_ys.col(0) = ys.col(i - 1);
        inner_lik.col(0) = lik.col(i - 1); //change 

        for (int j = 1; j < ti; ++j) {
            arma::vec noise(d);

            for (arma::uword l = 0; l < d; ++l) {
                noise[l] = R::rnorm(0.0, 1.0);
            }

            arma::vec gradient = 
                evaluate_grad_R(
                  grad, 
                  inner_xs.col(j - 1),
                  lam, 
                  dots, 
                  do_call
                );

            arma::mat sqrt_H =
                sHess_cpp(
                    inner_xs.col(j - 1),
                    lam,
                    A,
                    b
                );

            inner_ys.col(j) =
                inner_ys.col(j - 1) -
                eta * gradient / scale +
                std::sqrt(2.0 * eta / scale) *
                sqrt_H * noise;

            inner_xs.col(j) =
                inv_nabla_phi_cpp(
                    inner_ys.col(j),
                    lam,
                    non_diag
                );
            
            inner_lik.col(j) = log_E(inner_xs.col(j), X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(inner_xs.col(j)), 5000))+arma::sum(arma::exp(invvech_cpp(inner_xs.col(j)).diag()))) + log_prior(inner_xs.col(j)); 
        }

        xs.col(i) = inner_xs.col(ti - 1);
        ys.col(i) = inner_ys.col(ti - 1);
        lik.col(i) = inner_lik.col(ti - 1);

        all_x_blocks.push_back(inner_xs);
        all_lik_blocks.push_back(inner_lik); // change

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


// ============================================================
// Method to compare with MH step.
// ============================================================

// [[Rcpp::export]]
List DMLA_PGG_MH_order0(
    Function Fn,
    const arma::mat X,
    const arma::vec& x0,
    const int K,
    const double eta0,
    const double lam,
    const std::string grad_method,
    const double s0,
    const arma::mat& A,
    const arma::vec& b,
    const double scale,
    const bool message,
    const LogicalVector non_diag_r,
    List dots = List::create()

){
    RNGScope scope;
    dots["X"] = X;
    // Logistics to keep the function robust to user input.
    //////
    //////
    if (K < 1) {
        stop("K must be at least 1.");
    }
    if (eta0 <= 0.0) {
        stop("eta0 must be strictly positive.");
    }
    if (lam <= 0.0) {
        stop("lam must be strictly positive.");
    }
    if (scale <= 0.0) {
        stop("scale must be strictly positive.");
    }

    arma::uword d = x0.n_elem;

    if (A.n_cols != d) {
        stop("The number of columns of A must equal length(x0).");
    }
    if (A.n_rows != b.n_elem) {
        stop("The number of rows of A must equal length(b).");
    }
    if (static_cast<arma::uword>(non_diag_r.size()) != d) {
        stop("length(non_diag) must equal length(x0).");
    }

    arma::uvec non_diag(d);

    for (arma::uword i = 0; i < d; ++i) {
        if (LogicalVector::is_na(non_diag_r[i])) {
            stop("non_diag cannot contain NA.");
        }

        non_diag[i] = non_diag_r[i] ? 1u : 0u;
    }
    if (!is_feasible_grad_cpp(x0, A, b)) {
        stop("x0 must satisfy A * x0 - b < 0.");
    }
    //////
    //////
    // End logistics 

    bool use_centered;

    if (grad_method == "center" || grad_method == "centered") {
        use_centered = true;
    } else if (grad_method == "forward") {
        use_centered = false;
    } else {
        stop("grad_method must be 'center', 'centered', or 'forward'.");
    }

    double power = use_centered ? 1.0 / 6.0 : 1.0 / 4.0;
    double eta = eta0;

    arma::mat xs(d, K, arma::fill::zeros);
    arma::mat ys(d, K, arma::fill::zeros);
    arma::mat grads(d, K, arma::fill::zeros);
    arma::mat lik(1, K, arma::fill::zeros);
    xs.col(0) = x0;
    ys.col(0) = nabla_phi_cpp(x0, lam, A, b);
    lik.col(0) = log_E(x0, X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(x0), 5000)) + arma::sum(arma::exp(invvech_cpp(x0).diag()))) + log_prior(x0);

    Function do_call("do.call");

    double eps = s0 * std::pow(eta, power);
    arma::vec log_accept_vec = arma::zeros<arma::vec>(K);
    Rcpp::LogicalVector accept_vec(K);
    std::vector<HessInfo> all_hess_info(K);

    if (use_centered) {
            grads.col(0) = grad_est_centered_cpp(
                Fn,
                xs.col(0),
                eps,
                lam,
                A,
                b,
                non_diag,
                dots,
                do_call
            );
        } else {
            grads.col(0) = grad_est_forward_cpp(
                Fn,
                xs.col(0),
                eps,
                lam,
                A,
                b,
                non_diag,
                dots,
                do_call
            );
        }
    

    log_accept_vec[0] = 0;
    accept_vec[0] = true;
    all_hess_info[0] = HessInfo_impl(xs.col(0), lam, A, b);
    
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 1; i < K; ++i) {

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
            inv_nabla_phi_cpp(ys.col(i), lam, non_diag);

        lik.col(i) = log_E(xs.col(i), X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(xs.col(i)), 5000))+arma::sum(arma::exp(invvech_cpp(xs.col(i)).diag()))) + log_prior(xs.col(i)); 

        HessInfo hess_new_info = HessInfo_impl(xs.col(i), lam, A, b);
        all_hess_info[i] = hess_new_info;
        const arma::mat inv_hess_new = hess_new_info.invH;
        const double logdet_new = hess_new_info.logdetH;

        // The MH step 
        /////
        /////
        // 
        if (use_centered) {
            grads.col(i) = grad_est_centered_cpp(
                Fn,
                xs.col(i),
                eps,
                lam,
                A,
                b,
                non_diag,
                dots,
                do_call
            );
        } else {
            grads.col(i) = grad_est_forward_cpp(
                Fn,
                xs.col(i),
                eps,
                lam,
                A,
                b,
                non_diag,
                dots,
                do_call
            );
        }

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

            //Rcpp::Rcout << "log_frac_11:" <<  arma::dot(
            //        residual_old,
            //        inv_hess_old * residual_old
            //    ) / (4.0 * eta / scale) << "\n" << std::flush;
            
            //Rcpp::Rcout << "log_frac_12:" <<  arma::dot(
            //        residual_new,
            //        inv_hess_new * residual_new
            //    ) / (4.0 * eta / scale) << "\n" << std::flush;

            //Rcpp::Rcout << "log_frac_2:" << log_frac_2 << "\n" << std::flush;


            
            // Compute log posterior difference
            // Note that Fn is minus log posterior
            double log_posterior_diff = log_E(xs.col(i),X) - log_E(xs.col(i-1),X);
            

            //Rcpp::Rcout << "log_posterior_diff1:" << log_posterior_diff << "\n" << std::flush;

            arma::mat Y = PGMsim_cpp(X.n_rows,invvech_cpp(xs.col(i)),200);
            log_posterior_diff += log_E(xs.col(i-1),Y) - log_E(xs.col(i),Y);

            //Rcpp::Rcout << "log_posterior_diff2:" << log_posterior_diff << "\n" << std::flush;

            log_posterior_diff += log_prior(xs.col(i)) - log_prior(xs.col(i-1));            

            //Rcpp::Rcout << "log_posterior_diff3:" << log_posterior_diff << "\n" << std::flush;

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

// [[Rcpp::export]]
List DMLA_PGG_MH_order1(
    Function grad,
    const arma::vec& x0,
    const arma::mat& X,
    const int K,
    const double eta0,
    const double lam,
    const arma::mat& A,
    const arma::vec& b,
    const double scale,
    const bool message,
    const LogicalVector non_diag_r,
    const List& dots
) {
    RNGScope scope;

    if (K < 1) {
        stop("K must be at least 1.");
    }

    if (eta0 <= 0.0) {
        stop("eta0 must be strictly positive.");
    }

    if (lam <= 0.0) {
        stop("lam must be strictly positive.");
    }

    if (scale <= 0.0) {
        stop("scale must be strictly positive.");
    }

    const arma::uword d = x0.n_elem;

    if (A.n_cols != d) {
        stop(
            "The number of columns of A must equal length(x0)."
        );
    }

    if (A.n_rows != b.n_elem) {
        stop(
            "The number of rows of A must equal length(b)."
        );
    }

    if (static_cast<arma::uword>(non_diag_r.size()) != d) {
        stop("length(non_diag) must equal length(x0).");
    }

    arma::uvec non_diag(d);

    for (arma::uword i = 0; i < d; ++i) {
        if (LogicalVector::is_na(non_diag_r[i])) {
            stop("non_diag cannot contain NA.");
        }

        non_diag[i] = non_diag_r[i] ? 1u : 0u;
    }

    if (!is_feasible_grad_cpp(x0, A, b)) {
        stop("x0 must satisfy A * x0 - b < 0.");
    }

    arma::mat xs(d, K, arma::fill::zeros);
    arma::mat ys(d, K, arma::fill::zeros);
    arma::mat grads(d, K, arma::fill::zeros);
    arma::mat lik(1, K, arma::fill::zeros);

    arma::vec log_accept_vec = arma::zeros<arma::vec>(K);
    Rcpp::LogicalVector accept_vec(K);
    std::vector<HessInfo> all_hess_info(K);
    Function do_call("do.call");

    xs.col(0) = x0;
    ys.col(0) =
        nabla_phi_cpp(x0, lam, A, b);

    grads.col(0) = 
        evaluate_grad_R(
        grad, 
        x0,
        lam, 
        dots, 
        do_call
        );
    lik.col(0) = log_E(x0, X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(x0), 5000)) + arma::sum(arma::exp(invvech_cpp(x0).diag()))) + log_prior(x0);
    
    log_accept_vec[0] = 0;
    accept_vec[0] = true;
    all_hess_info[0] = HessInfo_impl(xs.col(0), lam, A, b);

    double eta = eta0;

    auto start_time =
        std::chrono::steady_clock::now();

    for (int i = 1; i < K; ++i) {

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
            inv_nabla_phi_cpp(ys.col(i), lam, non_diag);
        
        
        lik.col(i) = log_E(xs.col(i), X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(xs.col(i)), 5000))+arma::sum(arma::exp(invvech_cpp(xs.col(i)).diag()))) + log_prior(xs.col(i)); 

        HessInfo hess_new_info = HessInfo_impl(xs.col(i), lam, A, b);
        all_hess_info[i] = hess_new_info;
        const arma::mat inv_hess_new = hess_new_info.invH;
        const double logdet_new = hess_new_info.logdetH;

        // The MH step 
        /////
        /////
        // 
            grads.col(i) = evaluate_grad_R(
                grad, 
                xs.col(i),
                lam, 
                dots, 
                do_call
            );

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

            //Rcpp::Rcout << "log_frac_11:" <<  arma::dot(
            //        residual_old,
            //        inv_hess_old * residual_old
            //    ) / (4.0 * eta / scale) << "\n" << std::flush;
            
            //Rcpp::Rcout << "log_frac_12:" <<  arma::dot(
            //        residual_new,
            //        inv_hess_new * residual_new
            //    ) / (4.0 * eta / scale) << "\n" << std::flush;

            //Rcpp::Rcout << "log_frac_2:" << log_frac_2 << "\n" << std::flush;


            
            // Compute log posterior difference
            // Note that Fn is minus log posterior
            double log_posterior_diff = log_E(xs.col(i),X) - log_E(xs.col(i-1),X);
            

            //Rcpp::Rcout << "log_posterior_diff1:" << log_posterior_diff << "\n" << std::flush;

            arma::mat Y = PGMsim_cpp(X.n_rows,invvech_cpp(xs.col(i)),200);
            log_posterior_diff += log_E(xs.col(i-1),Y) - log_E(xs.col(i),Y);

            //Rcpp::Rcout << "log_posterior_diff2:" << log_posterior_diff << "\n" << std::flush;

            log_posterior_diff += log_prior(xs.col(i)) - log_prior(xs.col(i-1));            

            //Rcpp::Rcout << "log_posterior_diff3:" << log_posterior_diff << "\n" << std::flush;

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




// [[Rcpp::export]]
List DMLA_new_grad_cpp_one_loop(
    Function grad,
    const arma::vec& x0,
    const arma::mat& X,
    const int K,
    const double eta0,
    const double lam,
    const double multiplier,
    const arma::mat& A,
    const arma::vec& b,
    const double scale,
    const bool message,
    const LogicalVector non_diag_r,
    const List& dots
) {
    RNGScope scope;

    if (K < 1) {
        stop("K must be at least 1.");
    }

    if (eta0 <= 0.0) {
        stop("eta0 must be strictly positive.");
    }

    if (lam <= 0.0) {
        stop("lam must be strictly positive.");
    }

    if (multiplier <= 0.0) {
        stop("multiplier must be strictly positive.");
    }

    if (scale <= 0.0) {
        stop("scale must be strictly positive.");
    }

    const arma::uword d = x0.n_elem;

    if (A.n_cols != d) {
        stop(
            "The number of columns of A must equal length(x0)."
        );
    }

    if (A.n_rows != b.n_elem) {
        stop(
            "The number of rows of A must equal length(b)."
        );
    }

    if (static_cast<arma::uword>(non_diag_r.size()) != d) {
        stop("length(non_diag) must equal length(x0).");
    }

    arma::uvec non_diag(d);

    for (arma::uword i = 0; i < d; ++i) {
        if (LogicalVector::is_na(non_diag_r[i])) {
            stop("non_diag cannot contain NA.");
        }

        non_diag[i] = non_diag_r[i] ? 1u : 0u;
    }

    if (!is_feasible_grad_cpp(x0, A, b)) {
        stop("x0 must satisfy A * x0 - b < 0.");
    }

    arma::mat xs(d, K, arma::fill::zeros);
    arma::mat ys(d, K, arma::fill::zeros);
    arma::mat lik(1, K, arma::fill::zeros);

    xs.col(0) = x0;
    ys.col(0) =
        nabla_phi_cpp(x0, lam, A, b);
    lik.col(0) = log_E(x0, X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(x0), 5000)) + arma::sum(arma::exp(invvech_cpp(x0).diag()))) + log_prior(x0);

    Function do_call("do.call");

    auto start_time =
        std::chrono::steady_clock::now();

    for (int i = 1; i < K; ++i) {

        arma::vec noise(d);

        for (arma::uword l = 0; l < d; ++l) {
            noise[l] = R::rnorm(0.0, 1.0);
        }

        arma::vec gradient = 
            evaluate_grad_R(
                grad, 
                xs.col(i - 1),
                lam, 
                dots, 
                do_call
            );

        arma::mat sqrt_H =
            sHess_cpp(
                xs.col(i - 1),
                lam,
                A,
                b
            );

        ys.col(i) =
            ys.col(i - 1) -
            eta0 * gradient / scale +
            std::sqrt(2.0 * eta0 / scale) *
            sqrt_H * noise;

        xs.col(i) =
            inv_nabla_phi_cpp(
                ys.col(i),
                lam,
                non_diag
            );
        
        lik.col(i) = log_E(xs.col(i), X) - X.n_rows*(log(ztheta_cpp(invvech_cpp(xs.col(i)), 5000))+arma::sum(arma::exp(invvech_cpp(xs.col(i)).diag()))) + log_prior(xs.col(i)); 


        if (message && (i+1)%100 == 0) {
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

    arma::uword start_column = 0;

    return List::create(
        Named("samples") = xs,
        Named("dual_samples") = ys,
        Named("time") = elapsed_seconds,
        Named("log_likelihood") = lik
    );
}
