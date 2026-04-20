/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"


#define DIE(assertion, call_description)				\
	do {								\
		/**
		 * Block Logic: Conditional state branch.
		 * Invariant: The conditional branch maintains control flow invariants.
		 */
		if (assertion) {					\
			fprintf(stderr, "(%s, %d): ",			\
					__FILE__, __LINE__);		\
			perror(call_description);			\
			exit(EXIT_FAILURE);				\
		}							\
	} while (0)




 
 
/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *M;
	double *ATA;
	double *AB;
	double *ABBT;
	int i, j, k;

	M = (double *)calloc(N * N, sizeof(double));
	DIE(M == NULL, "calloc M");

	ATA = (double *)calloc(N * N, sizeof(double));
	DIE(ATA == NULL, "calloc ATA");

	AB = (double *)calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	ABBT = (double *)calloc(N * N, sizeof(double));
	DIE(ABBT == NULL, "calloc ABBT");
	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k++) {
				ABBT[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
        	/**
        	 * Block Logic: Iterative processing loop.
        	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
        	 */
        	for (j = 0; j < N; j++) {
           		 /**
           		  * Block Logic: Iterative processing loop.
           		  * Invariant: Maintains sequence integrity while progressing through the defined bounds.
           		  */
           		 for (k = 0; k < i + 1; k++) {
                		ATA[i * N + j] += A[k * N + i] * A[k * N + j];
        		}
    		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			M[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
		}
	}	

	free(ATA);
	free(AB);
	free(ABBT);

	return M;
}
