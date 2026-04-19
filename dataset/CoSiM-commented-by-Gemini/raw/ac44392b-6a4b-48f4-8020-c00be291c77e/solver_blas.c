/**
 * @raw/ac44392b-6a4b-48f4-8020-c00be291c77e/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A * (B * B^T) + A^T * A.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
  /**
   * Functional Utility: Initializes dynamically allocated matrix buffer.
   */
  double *bbt = calloc(N*N, sizeof(double));

  /**
   * Functional Utility: Computes the base product bbt = B * B^T natively.
   */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 1, bbt, N);

  /**
   * Functional Utility: Re-evaluates bbt = A * bbt targeting upper-triangular parameter.
   */
  cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, bbt, N);

  /**
   * Functional Utility: Alters A = A^T * A evaluating upper-triangular structure in-place.
   */
  cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);

  /**
   * Block Logic: Synthesizes final matrix summation computing bbt = bbt + A.
   */
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      bbt[N*i + j] += A[N*i + j];
    }
  }
  return bbt;
}
