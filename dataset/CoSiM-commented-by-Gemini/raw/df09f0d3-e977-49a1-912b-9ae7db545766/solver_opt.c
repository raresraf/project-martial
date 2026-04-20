/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"
#include <string.h>


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *At = (double*) calloc(N * N, sizeof(double));
	double *Bt = (double*) calloc(N * N, sizeof(double));
	register int i, j, k;

	double *C1 = (double*) calloc(N * N, sizeof(double));

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N; i++) {
		register double *matrix_lin = A + N * i;
		register double *matrix_col = At + i;

        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for(j = 0; j < N; j++) {
			*matrix_col = *matrix_lin;
			matrix_lin++;
			matrix_col += N;
        }
    }

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N; i++) {
		register double *matrix_lin = B + N * i;
		register double *matrix_col = Bt + i;

        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for(j = 0; j < N; j++) {
			*matrix_col = *matrix_lin;
			matrix_lin++;
			matrix_col += N;
        }
    }

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N; i++) {
	
	register double *pC = C1 + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	
	register double *orig_pA = A + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(j = 0; j < N; j++) {
			
			register double *pA = orig_pA + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double *pBt = Bt + j * N + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			
			register double suma = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = i; k < N; k++) {
				suma += *pA * *pBt; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pA++;
				pBt++;
			}

			*pC += suma;
			pC++;
		}
	}


	
	double *C2 = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N; i++) {
		register double *pC = C2 + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *orig_pC1 = C1 + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(j = 0; j < N; j++) {
			register double *pC1 = orig_pC1; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pB = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = 0; k < N; k++) {
				suma += *pC1 * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pC1++;
				pB++;
			}

			*pC += suma;
			pC++;
		}
	}

	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N; i++) {
		register double *pC = C2 + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *orig_At = At + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(j = 0; j < N; j++) {
			register double *pAt = orig_At; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pA = At + j * N ; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = 0; k <= i && k <= j; k++) {
				suma += *pAt * *pA; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pA++;
			}

			*pC += suma;
			pC++;
		}
	}

	free(At);
	free(Bt);
	free(C1);

	return C2;	
}
