#include <Rcpp.h>

#include <propr/interface/lrm.hpp>
#include <propr/interface/device_selector.hpp>
#include <propr/context.h>

#include <propr/kernels/cpu/dispatch/lrm.hpp>
#include <propr/kernels/cuda/dispatch/lrm.cuh>

using namespace Rcpp;
using namespace propr;


// [[Rcpp::export]]
NumericVector lrm(NumericMatrix &Y,
                  NumericMatrix &W,
                  bool weighted,
                  double a,
                  NumericMatrix Yfull,
                  NumericMatrix Wfull,
                  bool use_gpu) {

    int nfeats = Y.ncol();
    int N_pairs = nfeats * (nfeats - 1) / 2;
    NumericVector result_vec(N_pairs);

    if (is_gpu_backend() || use_gpu) {
        if (!R_IsNA(a)) {
            if (weighted) {
                 dispatch::cuda::lrm_alpha_weighted(result_vec, Y, W, a, Yfull, Wfull);
            } else {
                dispatch::cuda::lrm_alpha(result_vec, Y, a, Yfull);
            }
        } else {
            if (weighted) {
                dispatch::cuda::lrm_weighted(result_vec, Y, Wfull);
            } else {
                dispatch::cuda::lrm_basic(result_vec, Y);
            }
        }
    } else {
        dispatch::cpu::lrm(result_vec, Y, W, weighted, a, Yfull, Wfull);
    }
    return result_vec;

}