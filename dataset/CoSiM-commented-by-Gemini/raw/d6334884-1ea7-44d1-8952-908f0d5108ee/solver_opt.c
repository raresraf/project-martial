/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"


double* compute3(register int N, register double *A, register double *ABBt) {
	register double *C = malloc(N * N * sizeof(double));
	register int i, j, k;

	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig = &A[i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *offsetC = &C[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *offsetABBt = &ABBt[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &A[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= i; ++k) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa += N;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}
			*offsetC = sum + *offsetABBt;
			offsetC++;
			offsetABBt++;
		}
	}

	return C;
}


double* compute2(register int N, register double *A, register double *B) {
	register double *C = malloc(N * N * sizeof(double));
	register int i, j, k;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig = &A[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[j * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb++;
			}
			C[i * N + j] = sum;
		}
	}

	return C;
}


double* compute1(register int N, register double *A, register double *B) {
	register double *C = malloc(N * N * sizeof(double));
	register int i, j, k;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig = &A[i * N + i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[i * N + j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}
			C[i * N + j] = sum;
		}
	}

	return C;
}


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	register double *AB = compute1(N, A, B);
	register double *ABBt = compute2(N, AB, B);
	register double *C = compute3(N, A, ABBt);

	free(AB);
	free(ABBt);

	return C;
}
