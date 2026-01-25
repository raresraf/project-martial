
#include "utils.h"
#include <string.h>

#define MATRIX_TIMES_MATRIX_TRANSPOSE 1
#define MATRIX_TRANSPOSE_TIMES_MATRIX 0

void multiply_with_transpose_matrix(int N, double* A, double* C, int comute) {
	register int i, j, k;

	register double *orig_ptrA = A;
	register double *orig_ptrC = C;

	register double *root_ptrA;

	register double *ptrA1;
	register double *ptrA2;

	register double *ptrC1;
	register double *ptrC2;

	if (comute) {	
		for (i = 0; i < N; ++i) {
			ptrA2 = orig_ptrA;
			ptrC1 = orig_ptrC;
			ptrC2 = orig_ptrC;

			for (j = i; j < N; ++j) {
				ptrA1 = orig_ptrA;

				for (k = 0; k < N; ++k) {
					*ptrC1 += *ptrA1 * *ptrA2;

					++ptrA1;
					++ptrA2;
				}
				*ptrC2 = *ptrC1;

				++ptrC1;
				ptrC2 += N;
			}

			orig_ptrA += N;
			orig_ptrC += N + 1;
		}
	} else {	
		for (i = 0; i < N; ++i) {
			root_ptrA = orig_ptrA;

			for (k = 0; k < N; ++k) {
				ptrA1 = root_ptrA;
				ptrA2 = root_ptrA;
				ptrC1 = orig_ptrC;

				for (j = i; j < N; ++j) {
					*ptrC1 += *ptrA1 * *ptrA2;

					++ptrA2;
					++ptrC1;
				}

				root_ptrA += N;
			}

			ptrC1 = orig_ptrC + 1;
			ptrC2 = orig_ptrC + N;

			for (j = i + 1; j < N; ++j) {
				*ptrC2 = *ptrC1;

				++ptrC1;
				ptrC2 += N;
			}

			++orig_ptrA;
			orig_ptrC += N + 1;
		}
	}
}

void multiply_with_upper_triangular_matrix(int N, double* A, double* B, double* C) {
	register int i, j, k;

	register double *orig_ptrA = A;
	register double *orig_ptrC = C;

	register double *ptrA;
	register double *ptrB;
	register double *ptrC;

	for (i = 0; i < N; ++i) {
		ptrA = orig_ptrA;

		for (k = i; k < N; ++k) {
			ptrB = B + k * N;
			ptrC = orig_ptrC;

			for (j = 0; j < N; ++j) {
				*ptrC += *ptrA * *ptrB;

				++ptrB;
				++ptrC;
			}

			++ptrA;
		}

		orig_ptrA += N + 1;
		orig_ptrC += N;
	}
}

void add(int N, double* A, double* B, double* C) {
	register int i, j;
	register double *ptrA = A, *ptrB = B, *ptrC = C;

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			*ptrC = *ptrA + *ptrB;

			++ptrA;
			++ptrB;
			++ptrC;
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