
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	
	double *first_mul = calloc (N * N, sizeof(double));

	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));

	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));

	if (!third_mul)
		return NULL;

	double *res = malloc (N * N * sizeof(double));

	if (!res)
		return NULL;

	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				first_mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				second_mul[i * N + j] += first_mul[i * N + k] * B[j * N + k]; 
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				third_mul[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			res[i * N + j] = second_mul[i * N + j] + third_mul[i * N + j];
		}
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);

	return res;
}
