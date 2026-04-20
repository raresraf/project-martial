/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"


double *allocate_matrix(int N) {
	double *res = calloc(N * N, sizeof(double));
	return res;
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = allocate_matrix(N);
	double *AxB = allocate_matrix(N);
	double *AxBxBt = allocate_matrix(N);
	double *AtxA = allocate_matrix(N);
    int i, j, k;
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
            double sum = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = i; k < N; k++) {
                sum += (A[i* N + k] * B[k * N + j]);
            }
            AxB[i * N + j] = sum;
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
            double sum = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = 0; k < N; k++) {
                sum += (AxB[i * N + k] * B[j * N + k]);
            }
            AxBxBt[i * N + j] = sum;
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
            double sum = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = 0; k <= i; k++) {
                sum += (A[k * N + i] * A[k * N + j]);
            }
            AtxA[i * N + j] = sum;
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
			C[i * N + j] = AxBxBt[i * N + j] + AtxA[i * N + j];
		}
	}

	free(AxB);
	free(AxBxBt);
	free(AtxA);
	return C;
}

