
#include "utils.h"



void matrix_multiplication_with_superior(int N, double *A, double *B, double *C) {	
	int i, j, k;


	for (i = 0; i < N; i++) {
		double *pa = &A[i * N];
		double *pc = C + i * N;

		for (j = 0; j < N; j++) {
			register double x = 0;
			

			for (k = i; k < N; k++) {
				x += *(pa + k) * B[k * N + j];
			}
			*(pc + j) = x;
		}
	}
}

void matrix_multiplication_with_transpose(int N, double *A, double *B, double *C) {	
	int i, j, k;
	double *temp = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		double *pa = A + i * N;
		double *ptemp = temp + i * N;		

		for (j = 0; j < N; j++) {
			double *pb = B + j * N;
			register double x = 0;

			for (k = 0; k < N; k++) {
				x += *(pa + k) * *(pb + k);
			}

			*(ptemp + j) = x;
		}
	}

	for (i = 0; i < N; i++) {
		double *pc = C + i * N;
		double *ptemp = temp + i * N;		

		for (j = 0; j < N; j++) {
			*(pc + j) = *(ptemp + j);
		}
	}

	free(temp);
}

void matrix_multiplication_with_lower_upper(int N, double *A, double *C) {
	double *temp = calloc(N * N, sizeof(double));
	int i, j, k; 


	for (i = 0; i < N; i++) {
		double *ptemp = temp + i * N;
		
		for (j = 0; j < N; j++) {
			register double x = 0;

			register int del = (i < j) ? i : j;

			for (k = 0; k <= del; k++) {
			 	x += A[k * N + i] * A[k * N + j];
			}

			*(ptemp + j) = x;
		}
	}

	for (i = 0; i < N; i++) {
		double *pc = C + i * N;
		double *ptemp = temp + i * N;

		for (j = 0; j < N; j++) {
			*(pc + j) += *(ptemp + j);
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

