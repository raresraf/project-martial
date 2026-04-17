/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver.
 * Relies on highly tuned vendor libraries for maximum SIMD/cache utilization.
 * Performs C = AtA + ABBt efficiently.
 */

#include "utils.h"
#include <stdlib.h>
#include <string.h>
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
	double *C, *BBt;

	C = calloc(N * N, sizeof(*C));
	BBt = calloc(N * N, sizeof(*BBt));

	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if((C == NULL) || (BBt == NULL)) {
		return NULL;
	}
	
	memcpy(C, A, N * N * sizeof(*C));
	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 0.0, BBt, N);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, BBt, N);
	
	cblas_daxpy(N * N, 1.0, BBt, 1.0, C, 1);
		
	free(BBt);
	
	return C;
}
