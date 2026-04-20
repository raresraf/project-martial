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
	
	
	double *AB = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));
	int i, j, k, M;

	
	register double *pAB = AB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i)
	{
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j)
		{
			
			register double *pA = A + i * N + i;  /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pB = B + j + i * N;  /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			register double sum = 0.0;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k)
			{
				sum += *pA * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pA++;
				pB += N;
			}
			*pAB = sum;
			pAB++;
		}
	}

	
	register double *pC = C; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i)
	{
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j)
		{
			
			register double *pAB = AB + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pB_t = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			register double sum = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k)
			{
				sum += *pAB * *pB_t; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAB++;
				pB_t++;
			}
			
			*pC = sum;
			pC++;
		}
	}

	
    register double *pA_tA = C;  /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (i = 0; i < N; ++i)
    {
    	/**
    	 * Block Logic: Iterative processing loop.
    	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
    	 */
    	for (j = 0; j < N; j++) 
		{
			
			register double *pA_t = A + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
       		register double *pA = A + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
			/**
			 * Block Logic: Conditional state branch.
			 * Invariant: The conditional branch maintains control flow invariants.
			 */
			if (i >= j) {
                M = j;
            } else {
                M = i;
            }
		
			register double sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= M; ++k)
        	{
				
           		sum += *pA_t * *pA; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pA_t += N;
				pA += N;
        	}
        	*pA_tA += sum;
        	pA_tA++;
		}
	}                                                                                                                                                                                                                                                                                                                                                              
	
	free(AB);

	return C;
}
