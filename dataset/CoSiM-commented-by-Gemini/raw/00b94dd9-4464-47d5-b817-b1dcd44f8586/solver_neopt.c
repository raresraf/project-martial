/**
 * @file solver_neopt.c
 * @brief A non-optimized, naive implementation of a matrix solver.
 *
 * This file provides a basic, loop-based implementation for a matrix computation,
 * serving as a baseline or educational example. It does not use optimized libraries
 * like BLAS.
 */
#include "utils.h"
#include <string.h>




/**
 * @brief Computes the transpose of a square matrix.
 * @param M The input square matrix of size N*N.
 * @param N The dimension of the matrix.
 * @return A new matrix which is the transpose of M. The caller is responsible for freeing this memory.
 */
static double *get_transpose(double *M, int N)
{
	double *tr = calloc(N * N, sizeof(double));
	/**
	 * @brief This nested loop iterates through the matrix to perform the transposition.
	 * Invariant: After each inner loop, one row of the transpose matrix is correctly filled.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i];
		}
	}
	return tr;
}

/**
 * @brief Solves a matrix equation of the form: (A*B)*B^T + A^T*A using a naive, non-optimized approach.
 * @param N The dimension of the square matrices A and B.
 * @param A Input matrix A, assumed to be upper triangular.
 * @param B Input matrix B.
 * @return A new matrix containing the result of the computation. The caller is responsible for freeing this memory.
 *
 * @b Algorithm:
 * 1. Computes `A^T * A` using a naive triple-nested loop, assuming A is upper triangular,
 *    thus A^T is lower triangular. The loop for k is `k <= i`.
 * 2. Computes `A * B` using a naive triple-nested loop, taking advantage of A being upper triangular (`k` from `i` to `N-1`).
 * 3. Computes `(A * B) * B^T` using a standard triple-nested loop for matrix multiplication.
 * 4. Adds the results from step 1 and step 3 to produce the final matrix.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	
	// Computes second_mul = A^T * A.
	double *second_mul = calloc(N * N, sizeof(double));
	double *At = get_transpose(A, N);    
	
	/**
	 * @brief This block computes the product of A-transpose and A.
	 * The loop `k <= i` is an optimization based on the assumption that A^T is lower triangular (A is upper triangular).
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k <= i; ++k) {
				second_mul[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	double *first_mul = calloc(N * N, sizeof(double));
	double *first_mul_helper = calloc(N * N, sizeof(double));
	double *Bt = get_transpose(B, N);
	
	/**
	 * @brief This block computes the product of A and B.
	 * The loop `k >= i` is an optimization based on A being an upper triangular matrix.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				first_mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	// Computes first_mul_helper = first_mul * B^T, which is (A * B) * B^T.
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				first_mul_helper[i * N + j] += first_mul[i * N + k] * Bt[k * N + j];
			}
		}
	}

	memcpy(first_mul, first_mul_helper, N * N * sizeof(double));
	
	// Adds the two intermediate results: first_mul = first_mul_helper + second_mul.
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			first_mul[i * N + j] += second_mul[i * N + j];
		}
	}

	
	// Clean up all allocated temporary matrices.
	free(first_mul_helper);
	free(At);
	free(Bt);
	free(second_mul);

	return first_mul;
}