
#include "utils.h"
#define MIN(x, y) (((x) < (y)) ? (x) : (y))





double* my_transpose(int N, double* to_be_transposed) {
	int i, j;
	double *transpose = (double*) calloc(N * N, sizeof(double));
	
	for (i  = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			transpose[i * N + j] = to_be_transposed[j * N + i];
		}
	}
	return transpose;
}

double* my_solver(int N, double *A, double* B) {
	int i, j, k;
	int min = 0;

	printf("NEOPT SOLVER\n");
	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *At, *Bt;

	double *another_C = (double*) calloc(N * N, sizeof(double));
	if (another_C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *res_A = (double*) calloc(N * N, sizeof(double));
	if (res_A == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *res = (double*) calloc(N * N, sizeof(double));
	if (res == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	At = my_transpose(N, A);
	Bt = my_transpose(N, B);

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = 0.0;
			for (k = i; k < N; ++k) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			another_C[i * N + j] = 0.0;
			for (k = 0; k < N; ++k) {
				another_C[i * N + j] += C[i * N + k] * Bt[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			min = MIN(i, j);
			res_A[i * N + j] = 0.0;
			for(k = 0; k < min + 1; ++k) {
				res_A[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			res[i * N + j] = another_C[i * N + j] + res_A[i * N + j];
		}
	}
	free(res_A);
	free(another_C);
	free(At);
	free(Bt);
	free(C);
	return res;
}
