/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
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

    
    double *res = calloc(N * N, sizeof(double));
    double *AB = calloc(N * N, sizeof(double));
    double *ABB = calloc(N * N, sizeof(double));
    double *AA = calloc(N * N, sizeof(double));

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (int i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
		    double suma = 0.0;
		    /**
		     * Block Logic: Iterative processing loop.
		     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		     */
		    for (int k = 0; k < N; k++) {
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k >= i)
					suma += A[i * N + k] * B[k * N + j];
		    }
		    AB[i * N + j] = suma;
		}

	int k = 0;
	
	double *Bt = calloc(N * N, sizeof(double));

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(int i = 0; i < N; i++){
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++){
            Bt[k++] = B[j * N + i];
        }
    }

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k < N; k++) {
				suma += AB[i * N + k] * Bt[k * N + j];
			}
			ABB[i * N + j] = suma;
		}

	

	double *At = calloc(N * N, sizeof(double));

	k = 0;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int i = 0; i < N; i++){
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (int j = 0; j < N; j++){
            At[k++] = A[j * N + i];
        }
    }

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (int i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (int j = 0; j < N; j++) {
			double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (int k = 0; k < N; k++) {
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k <= i)
					suma += At[i * N + k] * A[k * N + j];
			}
			AA[i * N + j] = suma;
		}
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int i = 0; i < N * N; i++) {
		res[i] = AA[i] + ABB[i];
	}


	free(AA);
	free(AB);
	free(ABB);
	free(At);
	free(Bt);

	

	return res;
}
