/**
 * @raw/a2bbe46e-b5be-494e-a77f-355e18afec69/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Evaluates C = (A * B) * B^T + A^T * A employing modular functions natively binding BLAS interfaces.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */

#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"

int min(int i, int j)
{
	if (i < j)
		return i;

	return j;
}

double *alloc_matrix(int N)
{
	return (double *)calloc(N * N, sizeof(double));
}

void free_matrix(double *A)
{
	free(A);
}

/**
 * Functional Utility: Executes dtrmm converting input parameters producing res = A * B.
 */
double *multiply(int N, double *A, double *B)
{
	double *res = alloc_matrix(N);

	memcpy(res, B, N * N * sizeof(*res));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		res, N
	);
	
	return res;
}

/**
 * Functional Utility: Defines standard transposed multiplication implementing res = A * At^T.
 */
double *multiply_with_transpose_right(int N, double *A, double *At)
{
	double *res = alloc_matrix(N);

	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N,
		N, 1,
		A, N,
		At, N,
		0, res, N);

	return res;
}

/**
 * Functional Utility: Computes transposed product res = A^T * At implicitly resolving res = A^T * A.
 */
double *multiply_with_transpose_left(int N, double *A, double *At)
{
	double *res = alloc_matrix(N);

	cblas_dgemm(
		CblasRowMajor,
		CblasTrans,
		CblasNoTrans,
		N, N,
		N, 1,
		A, N,
		At, N,
		0, res, N);

	return res;
}

/**
 * Functional Utility: Combines array states implementing sequential vector addition defining res = A + B.
 */
double *add(int N, double *A, double *B)
{
	int i = 0;
	int j = 0;
	register double *line_A = NULL;
	register double *line_B = NULL;
	register double *line_res = NULL;

	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++) {
		line_A = &A[i * N];
		line_B = &B[i * N];
		line_res = &res[i * N];
		for (j = 0; j < N; j++)
			*(line_res + j) = *(line_A + j) + *(line_B + j);
	}
	
	return res;
}

/**
 * Functional Utility: Chains logical transformations integrating high-performance algorithms effectively producing complete model output.
 */
double* my_solver(int N, double *A, double *B) {
	double *AB = NULL;
	double *ABBt = NULL;
	double *AtA = NULL;
	double *C = NULL;

	AB = multiply(N, A, B);
	ABBt = multiply_with_transpose_right(N, AB, B);
	AtA = multiply_with_transpose_left(N, A, A);
	C = add(N, ABBt, AtA);

	free_matrix(AB);
	free_matrix(ABBt);
	free_matrix(AtA);

	return C;
}
