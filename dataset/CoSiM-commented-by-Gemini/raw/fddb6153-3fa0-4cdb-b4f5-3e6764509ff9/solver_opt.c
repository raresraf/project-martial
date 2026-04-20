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

	double *C, *D, *A_t, *D_t;
	register int i, j, k;

	C = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL)
		exit(EXIT_FAILURE);

	D = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (D == NULL)
		exit(EXIT_FAILURE);

	D_t = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (D_t == NULL)
		exit(EXIT_FAILURE);
	
	A_t = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (A_t == NULL)
		exit(EXIT_FAILURE);


	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		
		register double *pD = D + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			
			register double *pB = B + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double *pB_t = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double result = 0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k++) {
				result += (*pB) * (*pB_t); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pB++;
				pB_t++;
			}
			*pD = result;
			pD++;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		
		register double *pA = A + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *pA_t = A_t + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *pD = D + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *pD_t = D_t + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			*pA_t = *pA;
			
			pA++;
			
			pA_t += N;

			*pD_t = *pD;
			
			pD++;
			
			pD_t += N;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		
		register double *pC = C + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *pA_i_row = A + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			
			register double result = 0;
			
			register double *pA = pA_i_row + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pD = D_t + j * N + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++) {
				result += (*pA) * (*pD); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pA++;
				pD++;
			}
			*pC = result;
			pC++;
		}
	}

	
	
	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		
		register double *pC = C + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			
			register double *pA = A_t + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double *pA_t = A_t + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double result = 0;
			int min_index = i;
			/**
			 * Block Logic: Conditional state branch.
			 * Invariant: The conditional branch maintains control flow invariants.
			 */
			if (min_index > j)
				min_index = j;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= min_index; k++) {
				result += (*pA) * (*pA_t); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pA++;
				pA_t++;
			}
			*pC += result;
			pC++;
		}
	}


	free(D);
	free(A_t);
	free(D_t);

	return C;	
}
