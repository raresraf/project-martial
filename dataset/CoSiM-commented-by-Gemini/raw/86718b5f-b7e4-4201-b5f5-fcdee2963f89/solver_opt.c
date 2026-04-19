/**
 * @raw/86718b5f-b7e4-4201-b5f5-fcdee2963f89/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A^T * A + (A * B) * B^T.
 * Algorithm: Matrix multiplication employing register caching and loop index caching to minimize redundant offset calculations.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for storing intermediate computation matrices.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	/**
	 * Functional Utility: Allocates memory for storing A^T * A.
	 */
	double *result1 = malloc(N * N * sizeof(double));
	
	/**
	 * Block Logic: Computes result1 = A^T * A.
	 * Optimization: Employs the `ikj` loop nesting strategy to improve spatial locality of memory access patterns.
	 * Invariant: Precalculates the multiplicative strides within the outer loops to reduce index arithmetic overhead.
	 */
	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (k = 0; k < N; k++) {
			register int kn = k * N;
			for (j = 0; j < N; j++) {
				result1[in + j] += A[kn + i] * A[kn + j];
			}
		}
	}

	/**
	 * Functional Utility: Allocates zero-initialized memory for storing A * B.
	 */
	double *result2 = calloc(N * N, sizeof(double));
	
	/**
	 * Block Logic: Computes result2 = A * B.
	 * Optimization: Exploits the upper triangular nature of matrix A by initiating `k` at `i`.
	 * Localizes sum aggregation within a CPU register before committing to the main memory.
	 */
	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			for (k = i; k < N; k++) {
				sum += A[in + k] * B[k * N + j];
			}
			result2[in + j] = sum;
		}
	}

	/**
	 * Functional Utility: Allocates zero-initialized memory for storing (A * B) * B^T.
	 */
	double *result3 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Computes result3 = result2 * B^T.
	 * Optimization: Integrates register caching for accumulators and array index bases (`in`, `jn`).
	 */
	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register int jn = j * N;
			for (k = 0; k < N; k++) {
				sum += result2[in + k] * B[jn + k];
			}
			result3[in + j] = sum;
		}
	}

	/**
	 * Functional Utility: Allocates zero-initialized memory for the final summation matrix.
	 */
	double *resfinal = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Synthesizes final matrix by summing the partial evaluations.
	 * Invariant: Element-wise addition combining `result1` and `result3`.
	 */
	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (j = 0 ; j < N; j++) {
			resfinal[in + j] = result3[in + j] + result1[in + j];
		}
	}

	free(result1);
	free(result2);
	free(result3);
	printf("OPT SOLVER\n");
	return resfinal;	
}

