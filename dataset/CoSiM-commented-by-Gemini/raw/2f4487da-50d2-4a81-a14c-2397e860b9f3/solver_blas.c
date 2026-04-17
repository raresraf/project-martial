/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver.
 * Relies on highly tuned vendor libraries for maximum SIMD/cache utilization.
 * Performs C = AtA + ABBt efficiently.
 */

/*
 * Module: solver_blas.c
 * Purpose: High-level matrix solver utilizing BLAS optimized routines for matrix multiplication and triangular solve.
 * Path: @raw/2f4487da-50d2-4a81-a14c-2397e860b9f3/solver_blas.c
 */
#include "utils.h"
#include "cblas.h"

/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C = (double *)malloc(N * N * sizeof(double));
    double *D = (double *)malloc(N * N * sizeof(double));

    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
    	1.0, B, N, B, N, 0.0, C, N);
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
    	CblasNonUnit, N, N, 1.0, A, N, C, N);

    
    cblas_dcopy(N * N, A, 1, D, 1);
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
    	CblasNonUnit, N, N, 1.0, A, N, D, N);

    
    cblas_daxpy(N * N, 1.0, C, 1, D, 1);

    free(C);
	return D;
}
