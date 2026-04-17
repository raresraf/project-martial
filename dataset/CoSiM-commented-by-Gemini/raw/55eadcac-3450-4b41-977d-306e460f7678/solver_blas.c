/**
 * @file solver_blas.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	int i, j;

	double* C = calloc(N * N, sizeof(double));
	double* prod1 = calloc(N * N, sizeof(double));
	double* result = calloc(N * N, sizeof(double));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (C == NULL || prod1 == NULL || result == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }

	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (j = 0; j < N; j++) {
			C[i * N + j] = B[i * N + j];
			result[i * N + j] = A[i * N + j];
		}
	}

	
	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
				CblasNonUnit, N, N, 1.0, A, N, result, N);

	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1.0, C, N, B, N, 1.0, result, N);

	
	free(prod1);
	free(C);
	return result;
}