
#include "utils.h"


double* add(int N, double *A, double *B) {
	double *C = calloc(N * N, sizeof(double));
	int i, j;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += A[i * N + j] + B[i * N + j];
		}
	}

	return C;
}


double* compute3(int N, double *A) {
	double *C = calloc(N * N, sizeof(double));
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	return C;
}


double* compute2(int N, double *A, double *B) {
	double *C = calloc(N * N, sizeof(double));
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[j * N + k];
			}
		}
	}

	return C;
}


double* compute1(int N, double *A, double *B) {
	double *C = calloc(N * N, sizeof(double));
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	return C;
}


double* my_solver(int N, double *A, double* B) {
	double *AB = compute1(N, A, B);
	double *ABBt = compute2(N, AB, B);
	double *AtA = compute3(N, A);
	double *C = add(N, ABBt, AtA);

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
