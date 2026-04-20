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
	printf("NEOPT SOLVER\n");

    double *AB = calloc(N * N, sizeof(double));
    double *C = calloc(N * N, sizeof(double));
    int i, j, k;
    
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (i = 0; i < N; ++i)
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (j = 0; j < N; ++j) {
            /**
             * Block Logic: Conditional state branch.
             * Invariant: The conditional branch maintains control flow invariants.
             */
            if (i < j)
                /**
                 * Block Logic: Iterative processing loop.
                 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                 */
                for (k = 0; k <= i; ++k)
                    C[i * N + j] += A[k * N + i] * A[k * N + j];
            else 
                /**
                 * Block Logic: Iterative processing loop.
                 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                 */
                for (k = 0; k <= j; ++k)
                    C[i * N + j] += A[k * N + i] * A[k * N + j];
        }
    
    
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (i = 0; i < N; ++i)
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (j = 0; j < N; ++j) {
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = i; k < N; ++k)
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
        }
    
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (i = 0; i < N; ++i)
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (j = 0; j < N; ++j) {
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = 0; k < N; ++k)
                C[i * N + j] += AB[i * N + k] * B[j * N + k];
        }
    free(AB);
    return C;
	
}
