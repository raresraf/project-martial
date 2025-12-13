
#include "utils.h"
#define MAX(x, y) x > y ? x : y 
#define MIN(x, y) (x > y ? y : x)

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;
	double *result1 = malloc(N * N * sizeof(double));
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result1[i * N + j] = 0.0;
			for (k = i; k < N; k++) {
				result1[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	double *result2 = malloc(N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result2[i * N + j] = 0;
      		for (k = 0; k < N; k++) {
				result2[i * N + j] += result1[i * N + k] * B[j * N + k];
      		}
   		}
	}
	
	double *result3 = malloc(N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result3[i * N + j] = 0;
      		for (k = 0; k <= MIN(i, j); k++) {
				result3[i * N + j] += A[k * N + i] * A[k * N + j];
      		}
   		}
	}
	double *finalResult = malloc(N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			finalResult[i * N + j] = result2[i * N + j] + result3[i * N + j];
   		}
	}
	free(result1);
	free(result2);
	free(result3);
	return finalResult;
}
