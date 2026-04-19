/**
 * @raw/8890c616-fbd7-4a4e-98ec-ad5ad4a09f55/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = (A * B) * B^T + A^T * A using SGEMM derivatives.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */
#include <string.h>
#include <stdlib.h>

#include "utils.h"
#include "cblas.h"

/**
 * Functional Utility: Handles allocation of contiguous matrix buffers.
 */
void allocate(int N, double **C, double **AB, double **ABB_t,
			 double **A_tA) {
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = malloc(N * N * sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABB_t = malloc(N * N * sizeof(**ABB_t));
	if (NULL == *ABB_t)
		exit(EXIT_FAILURE);

	*A_tA = malloc(N * N * sizeof(**A_tA));
	if (NULL == *A_tA)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double *B) {
	double *C, *AB, *ABB_t, *A_tA;

	allocate(N, &C, &AB, &ABB_t, &A_tA);

	int i, j;

	/**
	 * Functional Utility: Calculates the standard dot product AB = A * B.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		N, N, N,
		1.0, A, N, B, N,
		0.0, AB, N
	);
	
	/**
	 * Functional Utility: Calculates the trailing partial product ABB_t = AB * B^T.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, AB, N, B, N,
		0.0, ABB_t, N
	);

	/**
	 * Functional Utility: Calculates A_tA = A^T * A.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasTrans,
		CblasNoTrans,
		N, N, N,
		1.0, A, N, A, N,
		0.0, A_tA, N
	);

	/**
	 * Block Logic: Fuses the intermediate matrices.
	 * Invariant: Aggregates computed sub-products into the final structure C.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
		}
	}

	free(AB);
	free(ABB_t);
	free(A_tA);

	return C;
}
