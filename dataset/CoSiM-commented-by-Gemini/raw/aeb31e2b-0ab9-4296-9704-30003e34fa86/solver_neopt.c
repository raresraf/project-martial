/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */
/**
 * @raw/aeb31e2b-0ab9-4296-9704-30003e34fa86/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Standard nested loops generating explicit transposition maps and partial multiplications via symmetric optimizations.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ representing intermediate target buffers.
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
	printf("NEOPT SOLVER\n");
	double *C, sum, *aux;
	int i, j, k;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
	 */
	C = (double *)malloc(N * N * sizeof(double));
	aux = (double *)malloc(N * N * sizeof(double));

	/**
	 * Block Logic: Computes the base product aux = A * B.
	 * Invariant: Operates independently of specific matrix bounds traversing the full NxN block structure, pruning over k >= i.
	 */
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
			aux[i * N + j] = 0;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				aux[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Evaluates trailing product aux * B^T defining the vector output inside C.
	 * Invariant: Extracts parameters directly reflecting matrix B transposition sequentially mapping indices.
	 */
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
			C[i * N + j] = 0;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				C[i * N + j] += aux[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Derives A^T * A internally setting the solution inside C.
	 * Invariant: Evaluates over the upper triangle taking advantage of symmetric products limiting the traversal logic bounds structurally.
	 */
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
			int lim;
			sum = 0;

			/**
			 * Block Logic: Conditional state branch.
			 * Invariant: The conditional branch maintains control flow invariants.
			 */
			if (i < j)
				lim = i;
			else
				lim = j;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= lim; ++k) {
				sum += A[k * N + i] * A[k * N + j];
			}

			C[i * N + j] += sum;
		}
	}

	free(aux);

	return C;
}
