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
double* my_solver(int N, double *A, double* B) {

	double* b_transpus = calloc(N * N, sizeof(double));
	double* a_transpus = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (int i = 0; i < N; ++i ){
       /**
        * Block Logic: Iterative processing loop.
        * Invariant: Maintains sequence integrity while progressing through the defined bounds.
        */
       for (int j = 0; j < N; ++j ){

          int index1 = i*N+j;

          int index2 = j*N+i;

          b_transpus[index2] = B[index1];
          a_transpus[index2] = A[index1];

       }
    }

    double* resultat = calloc(N * N, sizeof(double));

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (int i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (int k = 0; k < N; k++){
            	/**
            	 * Block Logic: Conditional state branch.
            	 * Invariant: The conditional branch maintains control flow invariants.
            	 */
            	if(k >= i){
                	sum = sum + A[i * N + k] * B[k * N + j];
                }
            }
            resultat[i * N + j] = sum;
        }
    }


    double* resultat2 = calloc(N * N, sizeof(double));
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (int i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (int k = 0; k < N; k++){
                sum = sum + resultat[i * N + k] * b_transpus[k * N + j];
            }
            resultat2[i * N + j] = sum;
        }
    }


    double* resultat3 = calloc(N * N, sizeof(double));
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (int i = 0; i < N; i++) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (int k = 0; k < N; k++){
            	/**
            	 * Block Logic: Conditional state branch.
            	 * Invariant: The conditional branch maintains control flow invariants.
            	 */
            	if(i >= k){
                	sum = sum + a_transpus[i * N + k] * A[k * N + j];
                }
            }
            resultat3[i * N + j] = sum;
        }
    }

    double *resultat4 = calloc(N * N, sizeof(double));

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(int i = 0; i < N * N; i++){
    	resultat4[i] = resultat3[i] + resultat2[i];
    }

    free(resultat3);
    free(resultat2);
    free(resultat);
    free(a_transpus);
    free(b_transpus);

	return resultat4;
}
