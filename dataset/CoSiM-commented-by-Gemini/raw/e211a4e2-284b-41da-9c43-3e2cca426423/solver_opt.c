/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
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
      double *C = malloc(N * N * sizeof(double));
      register double *pi, *pj, *p; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
      register int i, j, k;

      
      /**
       * Block Logic: Iterative processing loop.
       * Invariant: Maintains sequence integrity while progressing through the defined bounds.
       */
      for (i = 0; i < N; i++) {
            p = B + i * N;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (j = 0; j < N; j++) {
                  pi = p;
                  pj = B + j * N;
                  register double sum = 0;
                  /**
                   * Block Logic: Iterative processing loop.
                   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                   */
                  for (k = 0; k < N; k++) {
                        sum += (*pi) * (*pj); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
                        pi++;
                        pj++;
                  }
                  C[i * N + j] = sum;
            }
      }

      
      /**
       * Block Logic: Iterative processing loop.
       * Invariant: Maintains sequence integrity while progressing through the defined bounds.
       */
      for (i = 0; i < N; i++) {
            p = A + i * N + i;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (j = 0; j < N; j++) {
                  pi = p;
                  pj = C + N * i + j;
                  register double sum = 0;
                  /**
                   * Block Logic: Iterative processing loop.
                   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                   */
                  for (k = i; k < N; k++) {
                        sum += (*pi) * (*pj); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
                        pi++;
                        pj += N;
                  }
                  C[i * N + j] = sum;
            }
      }

      
      /**
       * Block Logic: Iterative processing loop.
       * Invariant: Maintains sequence integrity while progressing through the defined bounds.
       */
      for (i = 0; i < N; i++) {
            p = A + i;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (j = 0; j < N; j++) {
                  pi = p;
                  pj = A + j;
                  register double sum = 0;
                  /**
                   * Block Logic: Iterative processing loop.
                   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                   */
                  for (k = 0; k <= MIN(i, j); k++) {
                        sum += (*pi) * (*pj); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
                        pi += N;
                        pj += N;
                  }
                  C[i * N + j] += sum;
            }
      }
      return C;
}
