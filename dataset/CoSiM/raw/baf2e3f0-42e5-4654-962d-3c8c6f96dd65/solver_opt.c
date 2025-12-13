
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double* X = malloc(N * N * sizeof(double));
	if (X == NULL) {
		return NULL;
	}

	double* Y = malloc(N * N * sizeof(double));
	if (Y == NULL) {
		return NULL;
	}

	
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			X[i * N + j] = B[i + j * N];
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double* line = A + i * N + i;
			register double* column = X + j * N + i;
			for (register int k = i; k < N; ++k) {
				sum += *line * *column;
				++line;
				++column;
			}
			Y[i * N + j] = sum;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double* line = Y + i * N;
			register double* column = B + j * N;
			for (register int k = 0; k < N; ++k) {
				sum += *line * *column;
				++line;
				++column;
			}
			X[i * N + j] = sum;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			Y[i * N + j] = A[i + j * N];
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double* line = Y + i * N;
			register double* column = Y + j * N;
			for (register int k = 0; k <= i; ++k) {
				sum += *line * *column;
				++line;
				++column;
			}
			X[i * N + j] += sum;
		}
	}

	free(Y);
	return X;
}
