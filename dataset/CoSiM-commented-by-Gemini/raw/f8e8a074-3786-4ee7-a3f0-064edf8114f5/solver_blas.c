/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include <string.h>
#include <stdlib.h>
#include "utils.h"
#include "cblas.h"

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double *final = calloc(N * N, sizeof(*final));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (final == NULL) {
		exit(-1);
	}

	double *multiply = calloc(N * N, sizeof(*multiply));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (multiply == NULL) {
		exit(-1);
	}


	double *a_t = calloc(N * N, sizeof(*a_t));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (a_t == NULL) {
		exit(-1);
	}

	
	memcpy(final, B, N * N * sizeof(*B));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
				CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N,
				final, N);

	
	memcpy(multiply, final, N * N * sizeof(*final));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1.0, multiply, N, B, N, 0.0, final, N);

	
	memcpy(a_t, A, N * N * sizeof(*A));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
				CblasTrans, CblasNonUnit, N, N, 
				1.0, A, N, a_t, N);

	
	cblas_daxpy(N * N, 1, a_t, 1, final, 1);

	free(a_t);
	free(multiply);

	return final;
}
