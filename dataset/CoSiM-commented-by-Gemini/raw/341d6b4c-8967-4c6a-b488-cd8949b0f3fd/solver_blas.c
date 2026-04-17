/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver.
 * Relies on highly tuned vendor libraries for maximum SIMD/cache utilization.
 * Performs C = AtA + ABBt efficiently.
 */

#include <stdlib.h>
#include <string.h>

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
	double *C = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));

	
	memcpy(AB, B, N * N * sizeof(double));
	
	memcpy(C, A, N * N * sizeof(double));	

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);

	free(AB);
	
	return C;
}
