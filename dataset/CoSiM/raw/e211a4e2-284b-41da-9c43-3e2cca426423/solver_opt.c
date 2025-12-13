
#include "utils.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))


double* my_solver(int N, double *A, double* B) {
      double *C = malloc(N * N * sizeof(double));
      register double *pi, *pj, *p;
      register int i, j, k;

      
      for (i = 0; i < N; i++) {
            p = B + i * N;
            for (j = 0; j < N; j++) {
                  pi = p;
                  pj = B + j * N;
                  register double sum = 0;
                  for (k = 0; k < N; k++) {
                        sum += (*pi) * (*pj);
                        pi++;
                        pj++;
                  }
                  C[i * N + j] = sum;
            }
      }

      
      for (i = 0; i < N; i++) {
            p = A + i * N + i;
            for (j = 0; j < N; j++) {
                  pi = p;
                  pj = C + N * i + j;
                  register double sum = 0;
                  for (k = i; k < N; k++) {
                        sum += (*pi) * (*pj);
                        pi++;
                        pj += N;
                  }
                  C[i * N + j] = sum;
            }
      }

      
      for (i = 0; i < N; i++) {
            p = A + i;
            for (j = 0; j < N; j++) {
                  pi = p;
                  pj = A + j;
                  register double sum = 0;
                  for (k = 0; k <= MIN(i, j); k++) {
                        sum += (*pi) * (*pj);
                        pi += N;
                        pj += N;
                  }
                  C[i * N + j] += sum;
            }
      }
      return C;
}
