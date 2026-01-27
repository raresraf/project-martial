
/**
 * @file solver_neopt.c
 * @brief Implements a non-optimized matrix computation solver.
 *
 * This module defines a solver function that computes C = (A_upper * B) * B^T + (A^T * A)
 * using direct, non-optimized nested loops for matrix operations. This version serves
 * as a baseline for performance comparisons against optimized implementations.
 *
 * Algorithm: Naive nested-loop matrix multiplication with conditional element access.
 * Time Complexity: O(N^4) due to nested loops and conditional checks within the innermost loop.
 * Space Complexity: O(N^2) for intermediate matrix storage.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	// Functional Utility: Indicates the use of the non-optimized solver.
	printf("NEOPT SOLVER\n");
	// Block Logic: Allocates memory for the result matrix C and two intermediate matrices D and E.
	// Invariant: All allocated pointers must be checked for NULL to prevent dereferencing issues.
	double *C = calloc(N * N, sizeof(double));
	double *D = calloc(N * N, sizeof(double));
	double *E = calloc(N * N, sizeof(double));

	int i, j, k;

	// Block Logic: Error handling for memory allocation failures.
	// Pre-condition: Memory allocation attempts have just been made.
	// Invariant: If any allocation fails, print an error and return NULL.
	if(C == NULL || D == NULL || E == NULL) {
		printf("Eroare la alocare");
		return NULL;
	}

	/**
	 * Block Logic: Computes the intermediate matrix D, where D = A_upper * B.
	 * A_upper refers to A where elements A[i][k] are considered zero if i > k.
	 * Pre-condition: Matrices A and B are N x N.
	 * Invariant: D[i][j] contains the sum of A[i][k] * B[k][j] for k >= i.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0;
			for(k = 0; k < N; k++) {
				// Inline: Applies the upper triangular condition for matrix A.
				if(i <= k) {
					sum += A[i * N + k] * B[k * N + j];
				}
			}
			D[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Computes the intermediate matrix E, where E = D * B^T.
	 * Pre-condition: Matrix D is the result of the previous computation, B is the input matrix.
	 * Invariant: E[i][j] contains the sum of D[i][k] * B[j][k].
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0;
			for(k = 0; k < N; k++) {
				sum += D[i * N + k] * B[j * N + k];
			}
			E[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Computes the final matrix C, where C = E + A^T * A.
	 * A^T * A refers to the product where only elements A[k][i] and A[k][j] are considered
	 * based on specific conditions (k >= i or k <= j).
	 * Pre-condition: Matrix E is the result of the previous computation, A is the input matrix.
	 * Invariant: C[i][j] is the sum of E[i][j] and the specific sum from A^T * A.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0;
			for(k = 0; k < N; k++) {
				// Inline: Applies conditional access for the elements of matrix A.
				if(k >= i || k <= j) {
					sum += A[k * N + i] * A[k * N + j];
				}
			}
			C[i * N + j] = E[i * N + j] + sum;
		}
	}

	// Functional Utility: Frees memory allocated for intermediate matrices D and E.
	free(D);
	free(E);

	// Functional Utility: Returns the pointer to the result matrix C.
	return C;
}

