#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	
	double *result1 = malloc(N * N * sizeof(double));
	
	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (k = 0; k < N; k++) {
			register int kn = k * N;
			for (j = 0; j < N; j++) {
				result1[in + j] += A[kn + i] * A[kn + j];
			}
		}
	}

	double *result2 = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			for (k = i; k < N; k++) {
				sum += A[in + k] * B[k * N + j];
			}
			result2[in + j] = sum;
		}
	}

	double *result3 = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register int jn = j * N;
			for (k = 0; k < N; k++) {
				sum += result2[in + k] * B[jn + k];
			}
			result3[in + j] = sum;
		}
	}

	double *resfinal = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		register int in = i * N;
		for (j = 0 ; j < N; j++) {
			resfinal[in + j] = result3[in + j] + result1[in + j];
		}
	}

	free(result1);
	free(result2);
	free(result3);
	printf("OPT SOLVER\n");
	return resfinal;	
}

