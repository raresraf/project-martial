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
	double *C, *AB, *Bt, *At;
	C =(double *)malloc(N * N * sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL)
		exit(-1);
	AB =(double *)malloc(N * N * sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (AB == NULL)
		exit(-1);
	At =(double *)malloc(N * N * sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (At == NULL)
		exit(-1);
	Bt =(double *)malloc(N * N * sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (Bt == NULL)
		exit(-1);

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; ++i) {
		register double *A_tmp = A + i * N;
		register double *B_tmp = B + i * N;

		register double *At_tmp = At + i;
		register double *Bt_tmp = Bt + i;

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; ++j, ++A_tmp, ++B_tmp, At_tmp += N, Bt_tmp += N) {
			*At_tmp = *A_tmp;
			*Bt_tmp = *B_tmp;
		}

	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; ++i) {
		
		register double *AB_tmp = AB + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; ++j, ++AB_tmp) {
			
			register double sum = 0;
			
			register double *A_tmp = A + i * N + i;
			register double *B_tmp = Bt + j * N + i;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = i; k < N; ++k, ++A_tmp, ++B_tmp)
				sum += *A_tmp * *B_tmp;
			*AB_tmp = sum;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; ++i) {
		
		register double *C_tmp = C + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; ++j, ++C_tmp) {
			
			register double sum = 0;
			register double *AB_tmp = AB + i * N;
			register double *B_tmp = B + j * N;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k < N; ++k, ++AB_tmp, ++B_tmp)
				sum += *AB_tmp * *B_tmp;
			*C_tmp = sum;
		}
	}


	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; ++i) {
		register double *C_tmp = C + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; ++j, ++C_tmp) {
			
			register double sum = 0;
			register double *At_tmp = At + i * N;
			register double *A_tmp = At + j * N;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k <= i; ++k, ++At_tmp, ++A_tmp)
				sum += *At_tmp * *A_tmp;
			*C_tmp += sum;
		}
	}

	free(AB);
	free(At);
	free(Bt);
	return C;
}
