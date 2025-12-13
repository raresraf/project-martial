
#include "utils.h"
#include <string.h>

#define MATRIX_TIMES_MATRIX_TRANSPOSE 1
#define MATRIX_TRANSPOSE_TIMES_MATRIX 0

void multiply_with_transpose_matrix(int N, double* A, double* C, int comute) {
	int i, j, k;

	if (comute) {	
		for (i = 0; i < N; ++i) {
			for (j = i; j < N; ++j) {
				for (k = 0; k < N; ++k) {
					
					*(C + i * N + j) += *(A + i * N + k) * *(A + j * N + k);
				}
				*(C + j * N + i) = *(C + i * N + j);
			}
		}
	} else {	
		for (i = 0; i < N; ++i) {
			for (j = i; j < N; ++j) {
				for (k = 0; k < N; ++k) {
					
					*(C + i * N + j) += *(A + k * N + i) * *(A + k * N + j);
				}
				*(C + j * N + i) = *(C + i * N + j);
			}
		}
	}
}

void multiply_with_upper_triangular_matrix(int N, double* A, double* B, double* C) {
	int i, j, k;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				
				*(C + i * N + j) += *(A + i * N + k) * *(B + k * N + j);
			}
		}
	}
}

void add(int N, double* A, double* B, double* C) {
	int i, j;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			
			*(C + i * N + j) = *(A + i * N + j) + *(B + i * N + j);
		}
	}
}


double* my_solver(int N, double* A, double* B) {
	

	double* B_times_B_tr = calloc(N * N, sizeof(double));
	if (!B_times_B_tr)
		return NULL;
	
	double* A_tr_times_A = calloc(N * N, sizeof(double));
	if (!A_tr_times_A)
		return NULL;

	double* C = calloc(N * N, sizeof(double)); 
	if (!C)
		return NULL;

	
	multiply_with_transpose_matrix(N, B, B_times_B_tr,
		MATRIX_TIMES_MATRIX_TRANSPOSE);

	
	multiply_with_transpose_matrix(N, A, A_tr_times_A,
		MATRIX_TRANSPOSE_TIMES_MATRIX);

	
	memset(B, 0, N * N * sizeof(double));
	multiply_with_upper_triangular_matrix(N, A, B_times_B_tr, B);

	
	add(N, B, A_tr_times_A, C);

	free(B_times_B_tr);
	free(A_tr_times_A);

	return C;
}