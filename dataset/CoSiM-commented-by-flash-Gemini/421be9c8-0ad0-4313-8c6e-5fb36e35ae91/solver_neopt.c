
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Naive matrix solver implementation with manual property management.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + (A * B) * B^T
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate buffer AB and the result matrix C.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double *C, *AB;
	int i, j, k;

	// Logic: Zero-initialization of result and intermediate matrices.
	C = (double *)calloc(sizeof(double), N * N);
	AB = (double *)calloc(sizeof(double), N * N);

	/**
	 * Block Logic: Compute C = A^T * A.
	 * Optimization: Exploits upper triangularity of A by restricting the k-range.
	 * Logic: Handles transposition by accessing A[k][i] instead of A[i][k]. 
	 * The complex termination condition (k <= i && i < j) || (k <= j && i >= j) 
	 * identifies the non-zero region for the symmetric product of a triangular matrix.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; (k <= i && i < j) || (k <= j && i >= j); ++k) {
				C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final accumulation C = C + AB * B^T.
	 * Logic: Computes the product with B^T by accessing B[j][k].
	 * Invariant: result is accumulated directly into matrix C which already 
	 * holds the (A^T * A) component.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				C[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	free(AB);
	return C;
}
