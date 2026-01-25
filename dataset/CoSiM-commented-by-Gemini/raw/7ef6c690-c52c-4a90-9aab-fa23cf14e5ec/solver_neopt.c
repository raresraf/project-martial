
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C = malloc(N * N * sizeof(*C));
	double *aux = malloc(N * N * sizeof(*aux));
	int i, j, k;

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			aux[i * N + j] = 0;
			for (k = i; k < N; ++k) {
				aux[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = 0;
			for (k = 0; k < N; ++k) {
				C[i * N + j] += aux[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			aux[i * N + j] = 0;
			for (k = 0; k < i + 1; ++k) {
				aux[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);
	return C;
}
