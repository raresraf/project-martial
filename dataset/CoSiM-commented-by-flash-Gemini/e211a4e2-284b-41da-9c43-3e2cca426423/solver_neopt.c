
#include "utils.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))


double* my_solver(int N, double *A, double* B) {
      double *C = malloc(sizeof(double) * N * N);

      
      for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                  C[i * N + j] = 0;
                  for (int k = 0; k < N; k++) {
                        C[i * N + j] += B[i * N + k] * B[j * N + k];
                  }
            }
      }

      
      for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                  double value = 0;
                  for (int k = i; k < N; k++) {
                        value += A[i * N + k] * C[k * N + j];
                  }
                  C[i * N + j] = value;
            }
      }

      
      for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                  for (int k = 0; k <= MIN(i, j); k++) {
                        C[i * N + j] += A[k * N + i] * A[k * N + j];
                  }
            }
      }

	return C;
}
