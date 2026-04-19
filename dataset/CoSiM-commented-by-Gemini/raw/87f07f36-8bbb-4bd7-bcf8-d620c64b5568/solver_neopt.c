/**
 * @raw/87f07f36-8bbb-4bd7-bcf8-d620c64b5568/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = A * B * B^T + A^T * A.
 * Algorithm: Modular functions for each matrix multiplication stage using standard nested loops.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate accumulation matrices.
 */
#include "utils.h"

/**
 * Functional Utility: Computes the partial multiplication C = A * B.
 * It assumes A is an upper triangular matrix and optimizes the innermost loop boundaries accordingly.
 */
void matrix_multiplication_with_superior(int N, double *A, double *B, double *C) {	
	int i, j, k;

	/**
	 * Block Logic: Upper triangular matrix multiplication.
	 * Invariant: i <= k, hence the loop for k starts at i, skipping zero elements below the main diagonal.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double x = 0;

			for (k = i; k < N; k++) {
				x += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = x;
		}
	}
}

/**
 * Functional Utility: Multiplies a general matrix A with a transposed matrix B.
 * Effectively computes C = A * B^T.
 */
void matrix_multiplication_with_transpose(int N, double *A, double *B, double *C) {	
	int i, j, k;
	double *temp = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Performs dot product over rows of A and rows of B (acting as columns of B^T).
	 * Invariant: The index calculation [j * N + k] iterates across columns of B^T sequentially.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double x = 0;

			for (k = 0; k < N; k++) {
				x += A[i * N + k] * B[j * N + k];
			}
			temp[i * N + j] = x;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = temp[i * N + j];
		}
	}

	free(temp);
}


/**
 * Functional Utility: Computes A^T * A and accumulates the result into C.
 * Thus C = C + A^T * A.
 */
void matrix_multiplication_with_lower_upper(int N, double *A, double *C) {
	double *temp = calloc(N * N, sizeof(double));
	int i, j, k; 

	/**
	 * Block Logic: Computes A^T * A while accounting for A's upper triangular structure.
	 * Invariant: The summation bound `del = min(i, j)` safely ignores zeros below the diagonal.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			double x = 0;

			int del = (i < j) ? i : j;

			for (k = 0; k <= del; k++) {
				x += A[k * N + i] * A[k * N + j];
			}

			temp[i * N + j] = x;
		}
	}

	/**
	 * Block Logic: Aggregates the newly computed A^T * A with the existing matrix C.
	 * Invariant: C receives the final update for the entire problem.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += temp[i * N + j];
		}
	}

	free(temp);
}

void print_matrix(int N, double *a) {
	int i, j;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%f ", a[i + j]);
		}
		printf("\n");
	}
}

/**
 * Functional Utility: Entry point orchestrating the multi-stage mathematical evaluation.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(double));
	
	matrix_multiplication_with_superior(N, A, B, C);
	matrix_multiplication_with_transpose(N, C, B, C);
	matrix_multiplication_with_lower_upper(N, A, C);

	return C;
}
