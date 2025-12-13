

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
  
  double* bbt = calloc(N*N, sizeof(double));
  double* res = calloc(N*N, sizeof(double));
  
  for (register int i = 0; i < N; ++i) {
    double *orig_row_b = B + N*i;
    
    
    for (register int j = i; j < N; ++j) {
      register double *row_b = orig_row_b;
      register double *column_b = B + N*j;
      register double sum = 0.0;
      for (register int k = 0; k < N; ++k) {
        
        sum += *row_b * *column_b;
        row_b++;
        column_b++;
      }
      bbt[N*i + j] += sum;
      if(i!=j) bbt[N*j + i] += sum;
    }
  }


  
  for (register int i = 0; i < N; ++i) {
    double *orig_pa = A + N*i;
    for (register int j = 0; j < N; ++j) {
      register double *pa = orig_pa + i;
      register double *pb = bbt + N*i + j;
      register double sum = 0;
      for (register int k = i; k < N; ++k) {
        sum += *pa * *pb;
        pa++;
        pb += N;
      }
      res[N*i + j] += sum;
    }
  }

  
  for (register int i = 0; i < N; ++i) {
    
    
    for (register int j = i; j < N; ++j) {
      register double *pa = A + i;
      register double *pb = A + j;
      register double sum = 0;
      for (register int k = 0; k < N; ++k) {
          sum += *pa * *pb;
          pa += N;
          pb += N;
      }
      res[N*i + j] += sum;
      if (i != j) res[N*j + i] += sum;
    }
  }
  free(bbt);

  return res;
}

