
#include "utils.h"




double* transpose(int N, double* A) {
	double* At = malloc(N * N * sizeof(double));

	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			At[i*N + j] = A[j*N + i];
		}
	}
	return At;
}

double* inmultire(int N, double *A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	int i, j, k;
	for (i = 0; i < N ; i++) {
		for (j = 0; j < N ; j++)
			for (k = 0; k < N; k++)
				M[i*N + j] += A[i*N + k] * B[k*N + j];
	}

	return M;
}

double* inmultireSuperiorTriunghiulara(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	int i, j, k;
	for (i = 0; i < N ; i++) {
		for (j = 0; j < N ; j++)
			for (k = i; k < N; k++)
				M[i*N + j] += A[i*N + k] * B[k*N + j];
	}

	return M;
}

double* inmultireLowerXUpper(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	int i, j, k;
	for (i = 0; i < N ; i++) {
		for (j = 0; j < N ; j++) {
			for (k = 0; k < j + 1; k++)
				M[i*N + j] += A[i*N + k] * B[k*N + j];
		}
	}

	return M;
}

double* adunare(int N, double *A, double* B) {
	double* M = malloc(N * N * sizeof(double));

	int i, j;
	for (i = 0; i < N ; i++) {
		for (j = 0; j < N ; j++)
			M[i*N + j] = A[i*N + j] + B[i*N + j]; 
	}
	return M;
}

double* my_solver(int N, double *A, double* B) {
	double* AxB = inmultireSuperiorTriunghiulara(N, A, B);
	double* B_transpose = transpose(N, B);
	double* firstPart = inmultire(N, AxB, B_transpose);
	double* A_transpose = transpose(N, A);
	double* secondPart = inmultireLowerXUpper(N, A_transpose, A);

	double* C = adunare(N, firstPart, secondPart);

	free(AxB);
	free(firstPart);
	free(A_transpose);
	free(B_transpose);
	free(secondPart);

	return C;
}
