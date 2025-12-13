
#include "utils.h"





int min(int x, int y)
{
	if (x > y)
		return y;
	return x;
}


double *add_matrix(double *mat1, double *mat2, int N)
{
	register int i, j;
	double *mat_sum = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		register double sum = 0.0;
        for (j = 0; j < N; j++) {
		    sum = mat1[i * N + j] + mat2[i * N + j];
			mat_sum[i * N + j] = sum;
		}
	}

	return mat_sum;
}


double *multiply_simple_matrix(double *mat1, double *mat2, int N)
{
	register int i, j, k;
	double *mat_mult = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double row_cols_sum = 0.0;
			for (k = i; k < N; k++)
				row_cols_sum += mat1[i * N + k] * mat2[k * N + j];
			mat_mult[i * N + j] = row_cols_sum;
		}
	}

	return mat_mult;
}


double *multiply_matrix_transpose(double *mat1, double *mat2, int N)
{
	register int i, j, k;
	double *mat_mult = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double row_cols_sum = 0.0;
			for (k = 0; k < N; k++)
				row_cols_sum += mat1[i * N + k] * mat2[j * N + k];
			mat_mult[i * N + j] = row_cols_sum;
		}
	}

	return mat_mult;
}


double *multiply_transpose_matrix(double *mat1, double *mat2, int N)
{
	register int i, j, k, min_index;
	double *mat_mult = calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			min_index = min(i, j) + 1;
			register double row_cols_sum = 0.0;
			for (k = 0; k < min_index; k++)
				row_cols_sum += mat1[k * N + i] * mat2[k * N + j];
			mat_mult[i * N + j] = row_cols_sum;
		}
	}

	return mat_mult;
}

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *AB = NULL;
	double *ABBt = NULL;
	double *AtA = NULL;
	double *result = NULL;

	
	AB = multiply_simple_matrix(A, B, N);

	
	ABBt = multiply_matrix_transpose(AB, B, N);

	
	AtA = multiply_transpose_matrix(A, A, N);

	
	result = add_matrix(ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return result;}
