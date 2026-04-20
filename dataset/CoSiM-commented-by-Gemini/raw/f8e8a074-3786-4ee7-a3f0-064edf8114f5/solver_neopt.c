/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	double *a_t = calloc(N * N, sizeof(*a_t));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (a_t == NULL) {
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

	double *final_res = calloc(N * N , sizeof(*final_res));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (final_res == NULL) {
		exit(-1);
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++) {
				multiply[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0 ; k < N; k++) {
				final_res[i * N + j] += multiply[i * N + k] * B[j * N + k];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= i; k++) {
				a_t[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			final_res[i * N + j] += a_t[i * N + j];
		}
	}
	
	free(multiply);
	free(a_t);

	return final_res;
}
