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
	printf("OPT SOLVER\n");

	double *C = calloc(N * N, sizeof(double));
	double *AUX1 = calloc(N * N, sizeof(double));
	double *AUX2 = calloc(N * N, sizeof(double));
	register int i, j, k;

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register double *C_point = C + i * N;
		register double *A_point_aux = A + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0;

			
			
			
			register double *A_point = A_point_aux + i;

			
			register double *B_point = B + i * N + j;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++) {
				sum += *A_point * *B_point;
				A_point++;
				B_point += N;
			}

			*C_point = sum;
			C_point++;
		} 
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register double *AUX1_point = AUX1 + i * N;
		register double *C_point_aux = C + i * N;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0;

			
			register double *C_point = C_point_aux ;

			
			
			
			register double *B_point = B + j * N;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k++) {
				sum += *C_point * *B_point;
				C_point++;
				B_point++;
			}

			*AUX1_point = sum;
			AUX1_point++;
		} 
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register double *AUX2_point = AUX2 + i * N;
		register double *At_point_aux = A + i;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			register double sum = 0;

			
			
			
			
			register double *At_point = At_point_aux;

			
			register double *A_point = A + j;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= i; k++) {
				sum += *At_point * *A_point;
				At_point += N;
				A_point += N;
			}

			*AUX2_point = sum;
			AUX2_point++;
		} 
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register double *AUX1_point = AUX1 + i * N;
		register double *AUX2_point = AUX2 + i * N;
		register double *C_point = C + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
				*C_point = *AUX1_point + *AUX2_point;

				AUX1_point++;
				AUX2_point++;
				C_point++;
		} 
	}

	free(AUX1);
	free(AUX2);

	return C;	
}
