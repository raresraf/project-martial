/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	size_t i;
	size_t j;

	
	double *C = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL) {
		perror("Calloc C");
		exit(EXIT_FAILURE);
	}

	
	double *AB = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (AB == NULL) {
		perror("Calloc AB");
		exit(EXIT_FAILURE);
	}

	
	double *P1 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (P1 == NULL) {
		perror("Calloc P1");
		exit(EXIT_FAILURE);
	}

	
	double *P2 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (P2 == NULL) {
		perror("Calloc P2");
		exit(EXIT_FAILURE);
	}

	
	memcpy(AB, B, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, P1, N);

	
	memcpy(P2, A, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, P2, N);

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			C[i * N + j] = P1[i * N + j] + P2[i * N + j];
		}
	}

	free(AB);
	free(P1);
	free(P2);

	return C;
}
