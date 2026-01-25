

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
  double *bbt = calloc(N*N, sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 1, bbt, N);
  cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, bbt, N);
  cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      bbt[N*i + j] += A[N*i + j];
    }
  }
  return bbt;
}
