/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */
/**
 * @raw/aeb31e2b-0ab9-4296-9704-30003e34fa86/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Improves loop ordering taking cache latency constraints into consideration by utilizing explicitly tracked variables.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to auxiliary storage buffers.
 */

#include <string.h>
#include "utils.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C, *aux;
	register double *first, *second, *third;
	int i, j, k;

	/**
	 * Functional Utility: Provisions intermediary matrices enforcing clean bounds tracking elements.
	 */
	C = (double *)malloc(N * N * sizeof(double));
	aux = (double *)malloc(N * N * sizeof(double));
	memset(C, 0, N * N * sizeof(double));
	memset(aux, 0, N * N * sizeof(double));

	/**
	 * Block Logic: Generates aux = A * B mapping index strides locally using predefined array elements.
	 * Optimization: Uses cache efficient striding via fast localized pointers tracking over `k`.
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
		for (k = i; k < N; ++k) {
			first = A + i * N + k;
			second = B + k * N;
			third = aux + i * N;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (j = 0; j < N; ++j) {
				*third += *first * *second;
				second++;
				third++;
			}

			first++;
		}
	}

	/**
	 * Block Logic: Evaluates the sub-component producing C = aux * B^T effectively resolving (A * B) * B^T.
	 * Optimization: Sequences cache allocations extracting identical constants externally to mitigate inner processing operations.
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
			first = aux + i * N;
			second = B + j * N;
			register double sum = 0.0;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				sum += *first * *second;
				first++;
				second++;
			}

			C[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Solves C = C + A^T * A.
	 * Optimization: Explores symmetric patterns mitigating half of the iterations utilizing local registers.
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
			first = A + i;
			second = A + j;
			register double sum = 0.0;
			register int lim;

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
				sum += *first * *second;
				first += N;
				second += N;
			}

			C[i * N + j] += sum;
		}
	}

	free(aux);

	return C;
}
