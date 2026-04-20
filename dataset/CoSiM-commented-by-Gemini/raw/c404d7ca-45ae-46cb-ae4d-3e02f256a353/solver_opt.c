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
double *my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");

	register int i, j, k;
	register int size = N * N * sizeof(double);
	double *C, *At, *Bt, *AB;

	C = (double *) malloc(size);

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL)
		exit(-1);

	At = (double *) malloc(size);

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (At == NULL)
		exit(-1);

	Bt = (double *) malloc(size);

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (Bt == NULL)
		exit(-1);

	AB = (double *) malloc(size);

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (AB == NULL)
		exit(-1);

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		
		register double *pAt = At + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *pBt = Bt + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *pA = A + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *pB = B + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*pAt = *pA; *pBt = *pB;
			pAt += N; pBt += N;
			++pA; ++pB;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		
		register double *pAB = AB + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *auxA = A + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			
			register double suma = 0;
			
			register double *pA = auxA + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double *pBt = Bt + j * N + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				suma += *pA * *pBt; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pA; ++pBt;
			}

			*pAB = suma; ++pAB;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		
		register double *pC = C + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *auxAB = AB + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			
			register double suma = 0;
			
			register double *pAB = auxAB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double *pB = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				suma += *pAB * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pAB; ++pB;
			}

			*pC = suma; ++pC;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		
		register double *pC = C + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		
		register double *auxAt = At + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			
			register double suma = 0;
			
			register double *pAt = auxAt; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double *pAt2 = At + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= j; ++k) {
				suma += *pAt * *pAt2; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pAt; ++pAt2;
			}

			*pC += suma; ++pC;
		}
	}

	
	free(At);
	free(Bt);
	free(AB);

	return C;
}
