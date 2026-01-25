
#include "utils.h"


void matrix_mul_upper(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}


void matrix_mul_lower(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}


void matrix_mul(int N, double *A, double *B, double *C)
{
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}


void matrix_transpose(int N, double *A, double *AT)
{
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			AT[i * N + j] = A[j * N + i];
		}
	}
}


void matrix_add(int N, double *A, double *B, double *C)
{
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = A[i * N + j] + B[i * N + j];
		}
	}
}


double* my_solver(int N, double *A, double* B)
{
	
	double *AB = malloc(N * N * sizeof(double));
	matrix_mul_upper(N, A, B, AB);

	
	double *B_T = malloc(N * N * sizeof(double));
	matrix_transpose(N, B, B_T);

	
	double *ABB = malloc(N * N * sizeof(double));
	matrix_mul(N, AB, B_T, ABB);

	
	double *A_T = malloc(N * N * sizeof(double));
	matrix_transpose(N, A, A_T);

	
	double *AA = malloc(N * N * sizeof(double));
	matrix_mul_lower(N, A_T, A, AA);

	
	double *C = malloc(N * N * sizeof(double));
	matrix_add(N, ABB, AA, C);

	
	free(A_T);
	free(AB);
	free(B_T);
	free(ABB);
	free(AA);

	return C;
}
