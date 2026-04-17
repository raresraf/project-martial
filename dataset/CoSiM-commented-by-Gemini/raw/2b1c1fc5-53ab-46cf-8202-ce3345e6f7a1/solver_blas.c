/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver.
 * Relies on highly tuned vendor libraries for maximum SIMD/cache utilization.
 * Performs C = AtA + ABBt efficiently.
 */

#include "utils.h"
#include <string.h>
#include <stdlib.h>
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
	double *C;
	double *AA;

	C = malloc(N * N * sizeof(*C));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == C)
		exit(EXIT_FAILURE);

	AA = malloc(N * N * sizeof(*AA));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == AA)
		exit(EXIT_FAILURE);

	
	
	memcpy(C, B, N * N * sizeof(*C));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N
	);

	
	memcpy(AA, A, N * N * sizeof(*C));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AA, N
	);

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, C, N,
		B, N,
		1.0, AA, N
	);

	free(C);

	return AA;
}
