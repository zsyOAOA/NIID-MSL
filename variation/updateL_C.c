/*
 * =====================================================================================
 *
 *       Filename:  updateL_C.c
 *
 *    Description:  Update mogParam.L
 *                  prhs[0]: rau, V x 1 cell
 *                  prhs[1]: lambda, V x r matrix
 *                  prhs[2]: phiXk, V x 1 cell, T x 1
 *                  prhs[3]: resizeExRjj, r^2 x N
 *                  prhs[4]: R, r x N matrix
 *                  prhs[5]: noiseData, V x 1 cell, D x N
 *
 *                  plhs[0]: L_new, V x 1 cell, D x r matrix
 *                  plhs[1]: simgaLNew, V x 1 cell, r x r x D matrix
 *
 *        Version:  1.0
 *        Created:  12/11/18 19:02:48
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yue Zongsheng
 *   Organization:  XJTU
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mex.h>
#include <lapacke.h>

lapack_int inv_mat(double *A, unsigned int n);
mxArray *multiply_mat2(double *pt_X, const mwSize *pt_dimX, double *pt_Y, const mwSize *pt_dimY);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

    /*Variables declaration*/
    int jj, ii, nn, tt, rr1, offset_data, offset;
    mwIndex V, r, N, T, num_dim;
    const mwSize *Djj, *dim_R, *dim_resizeExRjj;
    mwSize dims_sigma[3];
    mxArray *LNew, *LNew_jj, *sigmaLNew, *sigmaLNew_jj, *rau_jj, *phiXk_jj;
    double *pt_LNew_jj, *pt_sigmaNewL_jj, *pt_rau_jj, *pt_phiXk_jj, *pt_sigmaNewL_jj_ii;
    mxArray *data_jj, *rauPhiXk, *rauPhiXkDotData;
    double *pt_data_jj, *pt_rauPhiXk, *pt_rauPhiXkDotData;
    mxArray *rauPhiXkRjj, *precisionVI, *L_jj_ii, *RrauPhiXkDotData;
    double *pt_rauPhiXkRjj, *pt_precisionVI, *pt_L_jj_ii, *pt_lambda;
    double *pt_resizeExRjj, *pt_R, add_inv = 1e-6;
    lapack_int ret;

    V = mxGetM(prhs[1]);
    num_dim = mxGetNumberOfDimensions(prhs[1]);
    if (num_dim == 1)
        r = 1;
    else
        r = mxGetN(prhs[1]);
    N =  mxGetN(prhs[4]);
    dims_sigma[0] = r;
    dims_sigma[1] = r;
    pt_lambda = mxGetPr(prhs[1]);
    pt_resizeExRjj = mxGetPr(prhs[3]);
    dim_resizeExRjj = mxGetDimensions(prhs[3]);
    pt_R = mxGetPr(prhs[4]);
    dim_R = mxGetDimensions(prhs[4]);
    if (nlhs != 2 || nrhs != 6){
        printf("The number of input or output is not right, num_in = 6, num_out = 2!");
        exit(EXIT_FAILURE);
    }
    LNew = mxCreateCellMatrix(V, 1);
    sigmaLNew = mxCreateCellMatrix(V, 1);
    for (jj = 0; jj < V; jj++){
        rau_jj = mxGetCell(prhs[0], jj);
        Djj = mxGetDimensions(rau_jj);  /* Djj: [D, N, T] */
        num_dim = mxGetNumberOfDimensions(rau_jj);
        if (num_dim == 2)
            T = 1;
        else
            T = *(Djj+2);
        pt_rau_jj = mxGetPr(rau_jj);
        phiXk_jj = mxGetCell(prhs[2], jj);
        pt_phiXk_jj = mxGetPr(phiXk_jj);
        dims_sigma[2] = *Djj;
        sigmaLNew_jj = mxCreateNumericArray(3, dims_sigma, mxDOUBLE_CLASS, mxREAL);
        pt_sigmaNewL_jj = mxGetPr(sigmaLNew_jj);
        LNew_jj = mxCreateDoubleMatrix(*Djj, r, mxREAL);
        pt_LNew_jj = mxGetPr(LNew_jj);
        data_jj = mxGetCell(prhs[5], jj);
        pt_data_jj = mxGetPr(data_jj);
        for (ii = 0; ii < *Djj; ii++){
            rauPhiXk = mxCreateDoubleMatrix(N, 1, mxREAL);
            pt_rauPhiXk = mxGetPr(rauPhiXk);
            rauPhiXkDotData = mxCreateDoubleMatrix(N, 1, mxREAL);
            pt_rauPhiXkDotData = mxGetPr(rauPhiXkDotData);
            for (nn = 0; nn < N; nn++){
                *(pt_rauPhiXk + nn) = 0;
                offset_data = ii + nn*(*Djj);
                for (tt = 0; tt < T; tt++){
                    offset = offset_data + tt*(N*(*Djj));
                    *(pt_rauPhiXk + nn) += *(pt_rau_jj + offset) * (*(pt_phiXk_jj + tt));
                }
                *(pt_rauPhiXkDotData + nn) = (*(pt_rauPhiXk + nn)) * (*(pt_data_jj \
                                                                            + offset_data));
            }
            rauPhiXkRjj = multiply_mat2(pt_resizeExRjj, dim_resizeExRjj, pt_rauPhiXk,\
                                                                mxGetDimensions(rauPhiXk));
            pt_rauPhiXkRjj = mxGetPr(rauPhiXkRjj);
            pt_sigmaNewL_jj_ii = pt_sigmaNewL_jj + ii*r*r;
            memcpy(pt_sigmaNewL_jj+ii*r*r, pt_rauPhiXkRjj, r*r*sizeof(double));
            for (rr1 = 0; rr1 < r;rr1++){
                offset = rr1 + rr1*r;
                *(pt_sigmaNewL_jj_ii + offset) += *(pt_lambda + jj + rr1*V);
            }
            ret = inv_mat(pt_sigmaNewL_jj_ii, r);
            /*while (ret != 0 && add_inv < 1e-3) {*/
                /*for (rr1 = 0; rr1 < r;rr1++){*/
                    /*offset = rr1 + rr1*r;*/
                    /**(pt_sigmaNewL_jj_ii + offset) += add_inv;*/
                /*}*/
                /*ret = inv_mat(pt_sigmaNewL_jj_ii, r);*/
                /*add_inv *= 10;*/
            /*}*/
            RrauPhiXkDotData = multiply_mat2(pt_R, dim_R,\
                                pt_rauPhiXkDotData, mxGetDimensions(rauPhiXkDotData));
            L_jj_ii = multiply_mat2(pt_sigmaNewL_jj_ii, dims_sigma, mxGetPr(RrauPhiXkDotData),\
                                                            mxGetDimensions(RrauPhiXkDotData));
            pt_L_jj_ii = mxGetPr(L_jj_ii);
            for (rr1 = 0; rr1 < r; rr1++){
                offset = ii + rr1 * (*Djj);
                *(pt_LNew_jj + offset) = *(pt_L_jj_ii + rr1);
            }
        }
        mxSetCell(LNew, jj, LNew_jj);
        mxSetCell(sigmaLNew, jj, sigmaLNew_jj);
    }
    plhs[0] = LNew;
    plhs[1] = sigmaLNew;
    mxDestroyArray(rauPhiXkRjj);
    mxDestroyArray(L_jj_ii); mxDestroyArray(precisionVI);
    mxDestroyArray(rauPhiXk); mxDestroyArray(rauPhiXkDotData);
    mxDestroyArray(RrauPhiXkDotData);
}

lapack_int inv_mat(double *A, unsigned int n){
    lapack_int ret, ipiv[n+1];
    ret = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, A, n, ipiv);
    if (ret != 0){
        printf("Failed to LU Decompose!");
        return ret;
    }
    ret = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, A, n, ipiv);
    if (ret != 0){
        printf("Failed to inverse matrix!");
    }
    return ret;
}

mxArray *multiply_mat2(double *pt_X, const mwSize *pt_dimX, double *pt_Y, const mwSize *pt_dimY){
    /*Input:*/
        /*X: m x k matrix*/
        /*dims_X: 2 x 1 array*/
        /*Y: k x n matrix*/
    mxArray *Z;
    double *pt_Z, temp;
    int ii, kk, jj, m, n, k, offset_X, offset_Y, offset_Z;

    if (*(pt_dimX+1) != *pt_dimY){
        printf(" Matrix dimensions don't match in function multiply_mat!\n");
        exit(EXIT_FAILURE);
        }
    else {
        m = *pt_dimX;
        k = *(pt_dimX + 1);
        n = *(pt_dimY + 1);
        }

    Z = mxCreateDoubleMatrix(m, n, mxREAL);
    pt_Z = mxGetPr(Z);

    for(ii = 0; ii < m; ii++){
        for (jj = 0; jj < n; jj++){
            temp = 0;
            for (kk = 0; kk < k; kk ++){
                offset_X = ii + kk*m;
                offset_Y = kk + jj*k;
                temp  += *(pt_X + offset_X) * (*(pt_Y + offset_Y));
            }
            offset_Z = ii + jj*m;
            *(pt_Z + offset_Z) = temp;
        }
    }
    return Z;
}
