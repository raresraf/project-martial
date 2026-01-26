
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
      double *C = malloc(sizeof(double) * N * N);
      
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, C, N);
      
      cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, C, N);
      
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 1, C, N);
      return C;
}
