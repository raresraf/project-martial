
#include "utils.h"

int mini(int a, int b) {
	return a < b ? a : b;
}

void multiply_AB(double *AB, double *A, double *B, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

void multiply_ABBt(double *ABBt, double *AB, double *B, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}
}

void multiply_AtA(double *AtA, double *A, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k <= mini(i, j); ++k) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
}

void addition(double *C, double *A, double *B, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			C[i * N + j] = A[i * N + j] + B[i * N + j];
		}
	}
}

double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(double));
	if (!C) {
		exit(-1);
	}

	double *AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(-1);
	}

	double *ABBt = calloc(N * N, sizeof(double));
	if (!ABBt) {
		exit(-1);
	}

	double *AtA = calloc(N * N, sizeof(double));
	if (!AtA) {
		exit(-1);
	}

	multiply_AB(AB, A, B, N);

	multiply_ABBt(ABBt, AB, B, N);

	multiply_AtA(AtA, A, N);

	addition(C, ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
