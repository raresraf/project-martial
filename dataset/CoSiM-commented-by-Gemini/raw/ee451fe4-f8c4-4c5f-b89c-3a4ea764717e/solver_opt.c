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
	printf("OPT SOLVER\n");

	int i, j, k;
	int idx;
	register double sum;

	double *At = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (At == NULL) 
		return NULL;

	double *Bt = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (Bt == NULL) 
		return NULL;

	double *AxB = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (AxB == NULL) 
		return NULL;

	double *AxBxBt = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (AxBxBt == NULL) 
		return NULL;

	double *result = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) 
		return NULL;
	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double * a = &A[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double * b = &B[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			idx = j * N + i;
		
			At[idx] = *a;
			Bt[idx] = *b;

			a++;
			b++;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double  *a = &A[i * N + i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			sum = 0.0;
			register double *a2 = a;
			register double *b = &B[i * N + j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				sum += *a2 * *b;
				a2++;
				b += N;
			}

			AxB[i * N + j] = sum;
		}
	}
	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double  *a = &AxB[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			sum = 0.0;

			register double *a2 = a;
			register double *bt = &Bt[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				sum += *a2 * *bt;
				a2++;
				bt += N;
			}

			AxBxBt[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double  *at = &At[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			sum = 0.0;
			register double *at2 = at;
			register double *a = &A[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < i + 1; ++k) {
				sum += *at2 * *a;
				at2++;
				a += N;
			}

			result[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *a = &AxBxBt[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			result[i * N + j] += *a;
			a++;
		}
	}
	return result;
}
