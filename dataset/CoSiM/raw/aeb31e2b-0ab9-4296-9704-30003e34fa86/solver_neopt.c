
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C, sum, *aux;
	int i, j, k;
	C = (double *)malloc(N * N * sizeof(double));
	aux = (double *)malloc(N * N * sizeof(double));

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
			int lim;
			sum = 0;

			if (i < j)
				lim = i;
			else
				lim = j;
			
			for (k = 0; k <= lim; ++k) {
				sum += A[k * N + i] * A[k * N + j];
			}

			C[i * N + j] += sum;
		}
	}

	free(aux);

	return C;
}
