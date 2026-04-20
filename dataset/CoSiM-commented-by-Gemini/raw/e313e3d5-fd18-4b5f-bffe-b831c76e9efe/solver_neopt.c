/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"

double* get_AxB(int N, double *A, double *B) {
	double *AxB = calloc(N * N, sizeof(double));
	int i, j, k;

	ASSERT(AxB == NULL, "malloc error");
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++)
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++)
				*(AxB + i * N + j) += *(A + i * N + k) * *(B + k * N + j);

	return AxB; 
}

double* get_AxBxBt(int N, double *A, double *B) {
	double *AxBxBt = calloc(N * N, sizeof(double));
	double *AxB = get_AxB(N, A, B);
	int i, j, k;

	ASSERT(AxBxBt == NULL, "malloc error");
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++)
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k++)
				*(AxBxBt + i * N + j) += *(AxB + i * N + k) * *(B + j * N + k);

	free(AxB);
	return AxBxBt;
}

double* get_AtxA(int N, double *A) {
	double *AtxA = calloc(N * N, sizeof(double));
	double *At = calloc(N * N, sizeof(double));
	int i, j, k;

	ASSERT(At == NULL, "malloc error");
	ASSERT(AtxA == NULL, "malloc error");
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = i; j < N; j++) {
			*(At + j * N + i) = *(A + i * N + j);
		}
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++)
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= j; k++)
				*(AtxA + i * N + j) += *(At + i * N + k) * *(A + k * N + j);

	free(At);
	return AtxA;
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *AxBxBt = get_AxBxBt(N, A, B);
	double *AxAt = get_AtxA(N, A);
	double *ret = malloc(N * N * sizeof(double));
	int i, j;

	ASSERT(ret == NULL, "malloc error");
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++)
			*(ret + i * N + j) = *(AxBxBt + i * N + j) + *(AxAt + i * N + j);

	free(AxBxBt);
	free(AxAt);
	return ret;
}
