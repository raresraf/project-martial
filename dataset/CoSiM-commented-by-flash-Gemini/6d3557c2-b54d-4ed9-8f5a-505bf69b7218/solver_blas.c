/**
 * @6d3557c2-b54d-4ed9-8f5a-505bf69b7218/solver_blas.c
 * @brief Optimized matrix expression solver using Level 3 BLAS.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) where A is upper 
 * triangular, using optimized TRMM and DGEMM operations to minimize FLOPs.
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>


/**
 * @brief Matrix solver kernel utilizing BLAS routine delegation.
 * Logic: Leverages triangular matrix structures to reduce dense computation overhead.
 */
double* my_solver(int N, double *A, double *B) {
	
	int i, j;

	// Pre-condition: Initialize intermediate buffer C with B for in-place TRMM.
	double *C = calloc(N * N, sizeof(double));
	memcpy(C, B, N * N * sizeof(double));

	/**
	 * Block Logic: Compute C = A * B.
	 * Optimization: Uses cblas_dtrmm to exploit the upper triangular property of A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
    

	double *D = calloc(N * N, sizeof(double));
	
	/**
	 * Block Logic: Compute D = C * B^T.
	 * Functional Utility: Dense GEMM for cross-product calculation.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, C, N, B, N, 0, D, N);

	// Pre-condition: Initialize E with A for in-place Gramian computation.
	double *E = calloc(N * N, sizeof(double));
	memcpy(E, A, N * N * sizeof(double));
	
	/**
	 * Block Logic: Compute E = A^T * A.
	 * Optimization: Efficiently calculates the dot products of A's columns in-place.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, E, N, E, N);

	/**
	 * Block Logic: Final result accumulation.
	 */
	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
			*(D + i * N + j) = *(D + i * N + j) + *(E + i * N + j);
        }
    }

	free(C);
	free(E);
	return D;
}
