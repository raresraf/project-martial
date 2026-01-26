
#include "utils.h"




void matrix_multiplication_with_superior(int N, double *A, double *B, double *C) {	
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double x = 0;

			for (k = i; k < N; k++) {
				x += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = x;
		}
	}
}


void matrix_multiplication_with_transpose(int N, double *A, double *B, double *C) {	
	int i, j, k;
	double *temp = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double x = 0;

			for (k = 0; k < N; k++) {
				x += A[i * N + k] * B[j * N + k];
			}
			temp[i * N + j] = x;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = temp[i * N + j];
		}
	}

	free(temp);
}



void matrix_multiplication_with_lower_upper(int N, double *A, double *C) {
	double *temp = calloc(N * N, sizeof(double));
	int i, j, k; 


	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double x = 0;

			int del = (i < j) ? i : j;

			for (k = 0; k <= del; k++) {
				x += A[k * N + i] * A[k * N + j];
			}

			temp[i * N + j] = x;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += temp[i * N + j];
		}
	}

	free(temp);
}

void print_matrix(int N, double *a) {
	int i, j;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%f ", a[i + j]);
		}
		printf("\n");
	}
}

double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(double));
	
	matrix_multiplication_with_superior(N, A, B, C);
	matrix_multiplication_with_transpose(N, C, B, C);
	matrix_multiplication_with_lower_upper(N, A, C);

	return C;
}
