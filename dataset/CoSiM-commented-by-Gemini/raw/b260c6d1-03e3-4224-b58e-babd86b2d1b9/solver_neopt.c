/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrices.
 */

#include "utils.h"




double* transpose(int N, double* A) {
	double* At = malloc(N * N * sizeof(double));

	int i, j;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			At[i*N + j] = A[j*N + i];
		}
	}
	return At;
}

double* inmultire(int N, double *A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	int i, j, k;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++)
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k++)
				M[i*N + j] += A[i*N + k] * B[k*N + j];
	}

	return M;
}

double* inmultireSuperiorTriunghiulara(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	int i, j, k;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++)
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++)
				M[i*N + j] += A[i*N + k] * B[k*N + j];
	}

	return M;
}

double* inmultireLowerXUpper(int N, double* A, double* B) {
	double* M = calloc(sizeof(double), N * N);

	int i, j, k;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < j + 1; k++)
				M[i*N + j] += A[i*N + k] * B[k*N + j];
		}
	}

	return M;
}

double* adunare(int N, double *A, double* B) {
	double* M = malloc(N * N * sizeof(double));

	int i, j;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N ; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N ; j++)
			M[i*N + j] = A[i*N + j] + B[i*N + j]; 
	}
	return M;
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double* AxB = inmultireSuperiorTriunghiulara(N, A, B);
	double* B_transpose = transpose(N, B);
	double* firstPart = inmultire(N, AxB, B_transpose);
	double* A_transpose = transpose(N, A);
	double* secondPart = inmultireLowerXUpper(N, A_transpose, A);

	double* C = adunare(N, firstPart, secondPart);

	free(AxB);
	free(firstPart);
	free(A_transpose);
	free(B_transpose);
	free(secondPart);

	return C;
}
