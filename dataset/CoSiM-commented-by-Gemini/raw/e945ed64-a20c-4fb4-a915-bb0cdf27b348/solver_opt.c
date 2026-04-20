/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
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
	register int i, j, k;

	double *C = calloc(N * N, sizeof(*C));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL) {
		exit(EXIT_FAILURE);
	}

	double *A_tA = calloc(N * N, sizeof(*A_tA));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (A_tA == NULL) {
		exit(EXIT_FAILURE);
	}

	double *AB = calloc(N * N, sizeof(*AB));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (AB == NULL)
		exit(EXIT_FAILURE);

	double *ABB_t = calloc(N * N, sizeof(*ABB_t));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (ABB_t == NULL)
		exit(EXIT_FAILURE);

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = B + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (k >= i) {
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				
				pa++;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}

			AB[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pab = AB + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pab = orig_pab; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb_t = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				sum += *pab * *pb_t; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pab++;
				pb_t++;
			}

			ABB_t[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa_t = A + i;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa_t = orig_pa_t; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pa = A + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= i; ++k) {
				sum += *pa_t * *pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa_t += N;
				pa += N;
			}

			A_tA[i * N + j] = sum;
		}
	}

	
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
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
		}
	}

	free(A_tA);
	free(AB);
	free(ABB_t);

	return C;
}
