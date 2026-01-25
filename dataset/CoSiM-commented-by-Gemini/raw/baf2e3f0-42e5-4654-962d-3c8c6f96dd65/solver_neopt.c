
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double* X = calloc(N * N, sizeof(double));
	if (X == NULL) {
		return NULL;
	}

	double* Y = calloc(N * N, sizeof(double));
	if (Y == NULL) {
		return NULL;
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				Y[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				X[i * N + j] += Y[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k <= i; ++k) {
				X[i * N + j] += A[i + k * N] * A[k * N + j];
			}
		}
	}

	free(Y);
	return X;
}
