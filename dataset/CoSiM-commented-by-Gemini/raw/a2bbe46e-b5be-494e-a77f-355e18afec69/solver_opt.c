/**
 * @raw/a2bbe46e-b5be-494e-a77f-355e18afec69/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Focuses on cache-locality and explicit loop optimizations via transposition, pointers, and CPU registers.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */

#include "utils.h"

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
 * Block Logic: Evaluates res = A * B utilizing variable pointers dynamically stored inside processor CPU registers.
 * Optimization: Isolates memory address lookups ensuring tight inner bounds targeting cache-optimized dimensions.
 */
double *multiply(int N, double *A, double *B)
{
	register int i = 0;
	register int j = 0;
	register int k = 0;
	register double *line_A = NULL;
	register double sum = 0.0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			line_A = &A[i * N];
			sum = 0.0;
			for (k = i; k < N; k++)
				sum += *(line_A + k) * B[k * N + j];
			res[i * N + j] = sum;
		}

	return res;
}

/**
 * Block Logic: Evaluates res = A * At^T utilizing variable pointers.
 */
double *multiply_with_transpose_right(int N, double *A, double *At)
{
	register int i = 0;
	register int j = 0;
	register int k = 0;
	register double sum = 0.0;
	register double *line_A = NULL;
	register double *line_At = NULL;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			line_A = &A[i * N];
			line_At = &At[j * N];
			sum = 0.0;
			for (k = 0; k < N; k++)
				sum += *(line_A + k) * *(line_At + k);
			res[i * N + j] = sum;
		}

	return res;
}

/**
 * Block Logic: Derives mapping symmetric elements evaluating res = At^T * A.
 * Optimization: Eliminates repetitive coordinate calculation copying elements over reflected boundary `res[i*N+j] = res[j*N+i]`.
 */
double *multiply_with_transpose_left(int N, double *A, double *At)
{
	register int i = 0;
	register int j = 0;
	register int k = 0;
	register double sum = 0.0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++)
		for (j = i; j < N; j++) {
			sum = 0.0;
			for (k = 0; k < min(i, j) + 1; k++)
				sum += At[k * N + i] * A[k * N + j];
			res[i * N + j] = sum;
			res[j * N + i] = sum;
		}

	return res;
}

/**
 * Functional Utility: Accelerates element-wise matrix addition utilizing contiguous pointer mapping.
 */
double *add(int N, double *A, double *B)
{
	register int i = 0;
	register int j = 0;
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
 * Functional Utility: Orchestrates the calculation sequence allocating explicit logic matrices sequentially.
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
