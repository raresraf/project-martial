
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *rez, *C;
	int i, j, k;

	C = calloc(N * N, sizeof(double));
	rez = calloc(N * N, sizeof(double));

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				rez[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				C[i * N + j] += A[i * N + k] * rez[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			rez[i * N + j] = 0;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= j; ++k) {
				rez[i * N + j] += A[k * N + i] * A[k * N + j];
			}
			C[i * N + j] += rez[i * N + j];
		}
	}
	
	free(rez);
	return C;
}
