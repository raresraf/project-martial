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
double* my_solver(int N, double *A, double *B) {
	
	double *C = (double *)malloc(N * N * sizeof(double));
	double *D = (double *)malloc(N * N * sizeof(double));

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
		double *ptrC = &(C[i * N]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			register double sum = 0.0;
			double *ptr = &(A[i * N + i]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			double *ptrB = &(B[i * N + j]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = i; k < N; k++) {
				sum += *ptr * (*ptrB); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				ptr++;
				ptrB += N;
			}
			*ptrC = sum;
			ptrC++;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
		double *ptrD = &(D[i * N]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			register double sum2 = 0.0;
			double *ptrC = &(C[i * N]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			double *ptrB = &(B[j * N]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k < N; k++) {
				sum2 += *ptrC * *ptrB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				ptrC++;
				ptrB++;
			}
			*ptrD = sum2;
			ptrD++;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++) {
		double *ptrC = &(C[i * (N + 1)]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = i; j < N; j++) {
			register double sum3 = 0.0;
			double *ptrA1 = &(A[j]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			double *ptrA2 = &(A[i]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k <= i; k++) {
				sum3 += *ptrA1 * (*ptrA2); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				ptrA1 += N;
				ptrA2 += N; 
			}
			*ptrC = sum3;
			ptrC++;
            /**
             * Block Logic: Conditional state branch.
             * Invariant: The conditional branch maintains control flow invariants.
             */
            if (i != j)
                C[j * N + i] = sum3;
		}
	}
	
	int size = N * N;
	double *ptrC = &(C[0]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	double *ptrD = &(D[0]); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < size; i++) {
		*ptrC += *ptrD;
		ptrC++;
		ptrD++;
	}

	free(D);

	return C;
}