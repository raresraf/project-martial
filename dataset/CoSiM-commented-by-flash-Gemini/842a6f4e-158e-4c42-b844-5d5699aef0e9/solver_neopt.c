
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	double *C = (double *)calloc(N * N, sizeof(double));
	if (C == NULL) return NULL;

	double *result1 = (double *)calloc(N * N, sizeof(double));
	if (result1 == NULL) return NULL;

	double *result2 = (double *)calloc(N * N, sizeof(double));
	if (result2 == NULL) return NULL;

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				if (i <= k) {
					result1[i * N + j] += A[i * N + k] * B[k * N + j];
				} else {
					continue;
				}
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				result2[i * N + j] += result1[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				if (k <= i) {
					C[i * N + j] += A[k * N + i] * A[k * N + j];
				} else {
					break;
				}
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += result2[i * N + j];
		}
	}

	free(result1);
	free(result2);

	return C;
}