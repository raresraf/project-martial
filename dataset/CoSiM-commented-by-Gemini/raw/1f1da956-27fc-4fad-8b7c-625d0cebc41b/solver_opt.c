/**
 * @file solver_opt.c
 * @brief A micro-optimized implementation of a matrix solver.
 * @details This file provides a solution for the matrix equation C = (A * B) * B' + A' * A,
 * where A is an upper triangular matrix. The implementation uses manual optimization
 * techniques such as the 'register' keyword and pointer arithmetic to improve performance
 * over the baseline version.
 */
#include "utils.h"

/**
 * @brief Returns the minimum of two integers.
 */
int mini(int a, int b) {
	return a < b ? a : b;
}

/**
 * @brief Computes the transpose of two matrices, A and B, simultaneously.
 * @param At Output matrix for the transpose of A.
 * @param Bt Output matrix for the transpose of B.
 * @param A First input matrix.
 * @param B Second input matrix.
 * @param N Dimension of the matrices.
 * @details This function uses pointer arithmetic and the 'register' keyword for optimization.
 * Time Complexity: O(N^2)
 */
void transpose(double *At, double *Bt, double *A, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		register double *A_line = A + i * N;
		register double *B_line = B + i * N;

		register double *At_col = At + i;
		register double *Bt_col = Bt + i;

		for (register int j = 0; j < N; ++j) {
			*At_col = *A_line;
			*Bt_col = *B_line;

			At_col += N;
			Bt_col += N;
			++A_line;
			++B_line;
		}
	}
}

/**
 * @brief Computes the matrix product AB = A * B using the transpose of B.
 * @param AB Output matrix to store the result.
 * @param A Input upper triangular matrix.
 * @param Bt Transpose of the input matrix B.
 * @param N Dimension of the matrices.
 * @details This function calculates the dot product of rows of A and columns of B (rows of Bt).
 * It uses pointer arithmetic and exploits the upper triangular nature of A.
 * Time Complexity: O(N^3), with a reduced operation count.
 */
void multiply_AB(double *AB, double *A, double *Bt, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double *Aptr = A + i * N + i;
			register double *Bptr = Bt + j * N + i;

			for (int k = i; k < N; ++k) {
				sum += (*Aptr) * (*Bptr);

				++Aptr;
				++Bptr;
			}
			AB[i * N + j] = sum;
		}
	}
}

/**
 * @brief Computes the matrix product ABBt = AB * B'.
 * @param ABBt Output matrix to store the result.
 * @param AB Input matrix (result of A * B).
 * @param B Input matrix.
 * @param N Dimension of the matrices.
 * @details This function calculates the dot product of rows of AB and columns of B' (rows of B).
 * It uses pointer arithmetic for optimized access.
 * Time Complexity: O(N^3)
 */
void multiply_ABBt(double *ABBt, double *AB, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double *Aptr = AB + i * N;
			register double *Bptr = B + j * N;

			for (register int k = 0; k < N; ++k) {
				sum += (*Aptr) * (*Bptr);
				++Aptr;
				++Bptr;
			}
			ABBt[i * N + j] = sum;
		}
	}
}

/**
 * @brief Computes the matrix product AtA = A' * A.
 * @param AtA Output matrix to store the result.
 * @param At Transpose of the input matrix A.
 * @param N Dimension of the matrix.
 * @details This function multiplies A' by A by taking the dot product of rows of A' (columns of A).
 * The loop `k <= mini(i, j)` exploits the sparsity of the operation.
 * Time Complexity: O(N^3), with a reduced operation count.
 */
void multiply_AtA(double *AtA, double *At, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double *Aptr = At + i * N;
			register double *Bptr = At + j * N;

			for (register int k = 0; k <= mini(i, j); ++k) {
				sum += (*Aptr) * (*Bptr);
				++Aptr;
				++Bptr;
			}
			AtA[i * N + j] = sum;
		}
	}
}

/**
 * @brief Performs element-wise addition of two matrices, C = A + B.
 * @details This version uses pointer arithmetic for sequential memory access.
 * Time Complexity: O(N^2)
 */
void addition(double *C, double *A, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			C[i * N + j] = *A + *B;
			++A;
			++B;
		}
	}
}

/**
 * @brief Solves C = (A * B) * B' + A' * A using a micro-optimized, modular approach.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details This function orchestrates the solver by calling a series of manually
 * optimized helper functions. It pre-computes transposes to potentially improve
 * access patterns in the multiplication routines.
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

	double *At = calloc(N * N, sizeof(double));
	if (!At) {
		exit(-1);
	}

	double *Bt = calloc(N * N, sizeof(double));
	if (!Bt) {
		exit(-1);
	}

	// Pre-compute transposes of A and B.
	transpose(At, Bt, A, B, N);

	// Step 1: Compute AB = A * B
	multiply_AB(AB, A, Bt, N);

	// Step 2: Compute ABBt = (A * B) * B'
	multiply_ABBt(ABBt, AB, B, N);

	// Step 3: Compute AtA = A' * A
	multiply_AtA(AtA, At, N);

	// Step 4: C = ABBt + AtA
	addition(C, ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);
	free(At);
	free(Bt);

	return C;
}
