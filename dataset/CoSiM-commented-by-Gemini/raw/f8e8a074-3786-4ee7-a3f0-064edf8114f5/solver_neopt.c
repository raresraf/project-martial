
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	double *a_t = calloc(N * N, sizeof(*a_t));
	if (a_t == NULL) {
		exit(-1);
	}

	double *multiply = calloc(N * N, sizeof(*multiply));
	if (multiply == NULL) {
		exit(-1);
	}

	double *final_res = calloc(N * N , sizeof(*final_res));
	if (final_res == NULL) {
		exit(-1);
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				multiply[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0 ; k < N; k++) {
				final_res[i * N + j] += multiply[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				a_t[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			final_res[i * N + j] += a_t[i * N + j];
		}
	}
	
	free(multiply);
	free(a_t);

	return final_res;
}
