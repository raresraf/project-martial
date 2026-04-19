/**
 * @raw/ac44392b-6a4b-48f4-8020-c00be291c77e/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = A * (B * B^T) + A^T * A.
 * Algorithm: Standard nested loops generating explicit transposition maps and partial multiplications via symmetric optimizations.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ representing intermediate target buffers.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
  
  /**
   * Functional Utility: Allocates memory for computing intermediate and final matrix equations.
   */
  double* bbt = calloc(N*N, sizeof(double));
  double* res = calloc(N*N, sizeof(double));
  
  /**
   * Block Logic: Computes bbt = B * B^T.
   * Invariant: Exploits symmetry of the resulting matrix; iterates across the upper triangle and mirrors it to the lower diagonal.
   */
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        
        sum += B[N*i + k] * B[N*j + k];
      }
      bbt[N*i + j] += sum;
      if(i!=j) bbt[N*j + i] += sum;
    }
  }


  /**
   * Block Logic: Computes A * (B * B^T).
   * Invariant: Respects the upper-triangular structure of A by restricting iteration (k >= i).
   */
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = i; k < N; ++k) {
        sum += A[N*i + k] * bbt[N*k + j];
      }
      res[N*i + j] += sum;
    }
  }

  /**
   * Block Logic: Evaluates A^T * A.
   * Invariant: Evaluates over the upper triangle taking advantage of symmetric products and mirroring.
   */
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < N; ++k) {
        sum += A[N*k + i] * A[N*k + j];
      }
      res[N*i + j] += sum;
      if (i != j) res[N*j + i] += sum;
    }
  }
  free(bbt);

  return res;
}
