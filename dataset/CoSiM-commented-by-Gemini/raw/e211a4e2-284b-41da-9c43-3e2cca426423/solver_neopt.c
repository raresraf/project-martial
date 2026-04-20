/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
      double *C = malloc(sizeof(double) * N * N);

      
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
                  C[i * N + j] = 0;
                  /**
                   * Block Logic: Iterative processing loop.
                   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                   */
                  for (int k = 0; k < N; k++) {
                        C[i * N + j] += B[i * N + k] * B[j * N + k];
                  }
            }
      }

      
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
                  double value = 0;
                  /**
                   * Block Logic: Iterative processing loop.
                   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                   */
                  for (int k = i; k < N; k++) {
                        value += A[i * N + k] * C[k * N + j];
                  }
                  C[i * N + j] = value;
            }
      }

      
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
                  /**
                   * Block Logic: Iterative processing loop.
                   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                   */
                  for (int k = 0; k <= MIN(i, j); k++) {
                        C[i * N + j] += A[k * N + i] * A[k * N + j];
                  }
            }
      }

	return C;
}
