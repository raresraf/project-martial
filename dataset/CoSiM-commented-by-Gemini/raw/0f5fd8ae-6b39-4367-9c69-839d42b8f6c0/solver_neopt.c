/**
 * @file solver_neopt.c
 * @brief A modular, non-optimized C implementation of a matrix solver.
 *
 * This file provides a `my_solver` implementation that computes the expression
 * `C = (A * B) * B^T + A^T * A`. The computation is broken down into separate
 * helper functions for each matrix multiplication and the final addition.
 * Each of these functions uses simple, non-optimized C for-loops.
 */
#include "utils.h"



/**
 * @brief Computes the matrix product C = A * B, assuming A is upper triangular.
 * @param N The dimension of the matrices.
 * @param A A pointer to the N x N upper triangular input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 */
double* AB(int N, double *A, double *B) {
	int i, j, k;
	double *AB;
	
	AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	return AB;
}


/**
 * @brief Computes the matrix product C = AB * B^T.
 * @param N The dimension of the matrices.
 * @param AB A pointer to the N x N input matrix AB.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 */
double* ABB_t(int N, double *AB, double *B) {
	int i, j, k;
	double *ABB_t;
	
	ABB_t = calloc(N * N, sizeof(double));
	if (!ABB_t) {
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];

	return ABB_t;
}


/**
 * @brief Computes the matrix product C = A^T * A.
 * @note The parameter A_t is passed A in my_solver, so this computes A^T * A.
 * @param N The dimension of the matrices.
 * @param A_t A pointer to the N x N input matrix A.
 * @param A A pointer to the N x N input matrix A.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 */
double* A_tA(int N, double *A_t, double *A) {
	int i, j, k;
	double *A_tA;
	
	A_tA = calloc(N * N, sizeof(double));
	if (!A_tA) {
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < i + 1; k++)
				A_tA[i * N + j] += A_t[k * N + i] * A[k * N + j];

	return A_tA;
}


/**
 * @brief Computes the element-wise sum of two matrices, C = A + B.
 * @param N The dimension of the matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the sum.
 */
double* sum(int N, double *A, double *B) {
	int i, j;
	double *C;
	
	C = calloc(N * N, sizeof(double));
	if (!C) {
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = A[i * N + j] + B[i * N + j];

	return C;
}

/**
 * @brief Computes C = (A * B) * B^T + A^T * A by calling modular helper functions.
 *
 * This function orchestrates the matrix computation by calling a sequence of helper
 * functions for each step of the expression.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *
 * @note The sequence of operations is:
 * 1. `ptr_AB = A * B`
 * 2. `ptr_ABB_t = ptr_AB * B^T`
 * 3. `ptr_A_tA = A^T * A`
 * 4. `ptr_sum = ptr_ABB_t + ptr_A_tA`
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER
");

	double *ptr_AB = AB(N, A, B);
	double *ptr_ABB_t = ABB_t(N, ptr_AB, B);
	double *ptr_A_tA = A_tA(N, A, A);
	double *ptr_sum = sum(N, ptr_ABB_t, ptr_A_tA);

	free(ptr_AB);
	free(ptr_ABB_t);
	free(ptr_A_tA);

	return ptr_sum;
}
