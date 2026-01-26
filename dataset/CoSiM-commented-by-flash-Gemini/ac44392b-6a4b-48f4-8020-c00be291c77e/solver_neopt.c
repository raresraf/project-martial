
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
  
  double* bbt = calloc(N*N, sizeof(double));
  double* res = calloc(N*N, sizeof(double));
  
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


  
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = i; k < N; ++k) {
        sum += A[N*i + k] * bbt[N*k + j];
      }
      res[N*i + j] += sum;
    }
  }

  
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
