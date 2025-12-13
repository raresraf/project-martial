
#include "utils.h"
#include <string.h>




static double *get_transpose(double *M, int N)
{
	double *tr = calloc(N * N, sizeof(double));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i];
		}
	}
	return tr;
}

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	
	double *second_mul = calloc(N * N, sizeof(double));
	double *At = get_transpose(A, N);	
		
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k <= i; ++k) {
				second_mul[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	double *first_mul = calloc(N * N, sizeof(double));
	double *first_mul_helper = calloc(N * N, sizeof(double));
	double *Bt = get_transpose(B, N);
	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				first_mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				first_mul_helper[i * N + j] += first_mul[i * N + k] * Bt[k * N + j];
			}
		}
	}

	memcpy(first_mul, first_mul_helper, N * N * sizeof(double));
	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			first_mul[i * N + j] += second_mul[i * N + j];
		}
	}

	
	free(first_mul_helper);
	free(At);
	free(Bt);
	free(second_mul);

	return first_mul;
}
