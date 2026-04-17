/**
 * @file solver_neopt.c
 * @brief A non-optimized, baseline implementation of a matrix solver.
 * @details This file provides a straightforward, unoptimized solution for the matrix equation
 * C = (A * B) * B' + A' * A, where A is an upper triangular matrix. The implementation
 * is broken down into several helper functions for clarity.
 */
#include "utils.h"

/**
 * @brief Returns the minimum of two integers.
 */
int mini(int a, int b) {
	return a < b ? a : b;
}

/**
 * @brief Computes the matrix product AB = A * B.
 * @param AB Output matrix to store the result.
 * @param A Input upper triangular matrix.
 * @param B Input matrix.
 * @param N Dimension of the matrices.
 * @details This function specifically multiplies an upper triangular matrix A
 * with a dense matrix B. The inner loop starts from `k = i` to exploit the
 * zero elements in the lower part of A.
 * Time Complexity: O(N^3), but with a lower constant factor.
 */
void multiply_AB(double *AB, double *A, double *B, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

/**
 * @brief Computes the matrix product ABBt = AB * B'.
 * @param ABBt Output matrix to store the result.
 * @param AB Input matrix (result of A * B).
 * @param B Input matrix.
 * @param N Dimension of the matrices.
 * @details This function performs a matrix multiplication where the second matrix
 * is transposed. The access `B[j * N + k]` is equivalent to `B_transpose[k * N + j]`.
 * Time Complexity: O(N^3)
 */
void multiply_ABBt(double *ABBt, double *AB, double *B, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}
}

/**
 * @brief Computes the matrix product AtA = A' * A.
 * @param AtA Output matrix to store the result.
 * @param A Input upper triangular matrix.
 * @param N Dimension of the matrix.
 * @details This function multiplies the transpose of an upper triangular matrix A
 * with itself. The access `A[k * N + i]` is equivalent to `A_transpose[i * N + k]`.
 * The inner loop `k <= mini(i, j)` exploits the zero entries in both the
 * lower triangular A' and the upper triangular A.
 * Time Complexity: O(N^3), with a lower constant factor.
 */
void multiply_AtA(double *AtA, double *A, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k <= mini(i, j); ++k) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
}

/**
 * @brief Performs element-wise addition of two matrices, C = A + B.
 * @param C Output matrix.
 * @param A First input matrix.
 * @param B Second input matrix.
 * @param N Dimension of the matrices.
 * Time Complexity: O(N^2)
 */
void addition(double *C, double *A, double *B, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			C[i * N + j] = A[i * N + j] + B[i * N + j];
		}
	}
}

/**
 * @brief Solves the matrix equation C = (A * B) * B' + A' * A using a non-optimized, modular approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details The function allocates memory for intermediate results and calls helper
 * functions for each distinct matrix operation.
 * 1. Computes the intermediate product AB = A * B.
 * 2. Computes the first term ABBt = AB * B'.
 * 3. Computes the second term AtA = A' * A.
 * 4. Computes the final result C by adding the two terms.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(double));
	if (!C) {
		exit(-1);
	}

	double *AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(-1);
	}

	double *ABBt = calloc(N * N, sizeof(double));
	if (!ABBt) {
		exit(-1);
	}

	double *AtA = calloc(N * N, sizeof(double));
	if (!AtA) {
		exit(-1);
	}
	
	// Step 1: Compute AB = A * B
	multiply_AB(AB, A, B, N);

	// Step 2: Compute ABBt = (A * B) * B'
	multiply_ABBt(ABBt, AB, B, N);

	// Step 3: Compute AtA = A' * A
	multiply_AtA(AtA, A, N);

	// Step 4: C = ABBt + AtA
	addition(C, ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
