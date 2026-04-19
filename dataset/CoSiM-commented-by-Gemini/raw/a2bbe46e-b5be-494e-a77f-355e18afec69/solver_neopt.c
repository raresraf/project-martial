/**
 * @raw/a2bbe46e-b5be-494e-a77f-355e18afec69/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Modularized unoptimized routines iterating through standard nested loops.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for dynamic memory allocations.
 */

#include "utils.h"

/**
 * Inline: Utility calculating inclusive bounds bounds.
 */
int min(int i, int j)
{
	if (i < j)
		return i;

	return j;
}

/**
 * Functional Utility: Automates heap allocations for intermediate matrices initialized to zero.
 */
double *alloc_matrix(int N)
{
	return (double *)calloc(N * N, sizeof(double));
}

void free_matrix(double *A)
{
	free(A);
}

/**
 * Block Logic: Evaluates standard matrix dot product generating res = A * B.
 * Invariant: Enforces sparsity condition targeting upper triangular bounds `k >= i`.
 */
double *multiply(int N, double *A, double *B)
{
	int i = 0;
	int j = 0;
	int k = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++)
				res[i * N + j] += A[i * N + k] * B[k * N + j];
		}
	}

	return res;
}

/**
 * Block Logic: Derives cross-matrix operation calculating res = A * At^T.
 * Invariant: Reverses column and row dimensions via indexing simulating `A * At^T` structurally.
 */
double *multiply_with_transpose_right(int N, double *A, double *At)
{
	int i = 0;
	int j = 0;
	int k = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				res[i * N + j] += A[i * N + k] * At[j * N + k];

	return res;
}

/**
 * Block Logic: Formulates left-aligned transpose product res = At^T * A.
 * Invariant: Reduces bounds up to `min(i, j) + 1` anticipating symmetric configurations.
 */
double *multiply_with_transpose_left(int N, double *A, double *At)
{
	int i = 0;
	int j = 0;
	int k = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++)
			for (k = 0; k < min(i, j) + 1; k++)
				res[i * N + j] += At[k * N + i] * A[k * N + j];
	}

	return res;
}

/**
 * Block Logic: Aggregates arrays performing sequential element-wise sum defining res = A + B.
 */
double *add(int N, double *A, double *B)
{
	int  i = 0;
	int j = 0;
	double *res = alloc_matrix(N);

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			res[i * N + j] = A[i * N + j] + B[i * N + j];
	
	return res;
}

/**
 * Functional Utility: Coordinates sequence combining localized memory boundaries producing the comprehensive algorithm.
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
