
#include "utils.h"


static void matrix_mul_upper(register int N, register double *A, register double *B, register double *C)
{
	register int i, j, k;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (k = i; k < N; k++) {
			register double pa = A[line + k];
			register int k_line = k * N;

			for (j = 0; j < N; j++) {
				C[line + j] += pa * B[k_line + j];
			}
		}
	}
}


static void matrix_mul_lower(register int N, register double *A, register double *B, register double *C)
{
	register int i, j, k;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (k = 0; k <= i; k++) {
			register double pa = A[line + k];
			register int k_line = k * N;

			for (j = 0; j < N; j++) {
				C[line + j] += pa * B[k_line + j];
			}
		}
	}
}


static void matrix_mul(register int N, register double *A, register double *B, register double *C)
{
	register int i, j, k;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (k = 0; k < N; k++) {
			register double pa = A[line + k];
			register int k_line = k * N;

			for (j = 0; j < N; j++) {
				C[line+ j] += pa * B[k_line + j];
			}
		}
	}
}


static void matrix_transpose(register int N, register double *A, register double *AT)
{
	register int i, j;
	for (i = 0; i < N; i++) {
		register int line = i * N;

		for (j = 0; j < N; j++) {
			AT[line + j] = A[j * N + i];
		}
	}
}


static void matrix_add(register int N, register double *A, register double *B, register double *C)
{
	register int i;
	for (i = 0; i < N * N; i++) {
		C[i] = A[i] + B[i];
	}
}


double* my_solver(int N, double *A, double* B) {
	register int size = N * N * sizeof(double);
	
	register double *A_T = malloc(size);
	matrix_transpose(N, A, A_T);

	
	register double *AB = malloc(size);
	matrix_mul_upper(N, A, B, AB);

	
	register double *B_T = malloc(size);
	matrix_transpose(N, B, B_T);

	
	register double *ABB = malloc(size);
	matrix_mul(N, AB, B_T, ABB);

	
	register double *AA = malloc(size);
	matrix_mul_lower(N, A_T, A, AA);

	
	register double *C = malloc(size);
	matrix_add(N, ABB, AA, C);

	
	free(A_T);
	free(AB);
	free(B_T);
	free(ABB);
	free(AA);

	return C;
}
