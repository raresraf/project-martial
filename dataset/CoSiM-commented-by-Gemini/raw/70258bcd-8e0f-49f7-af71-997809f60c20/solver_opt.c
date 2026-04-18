
/**
 * @file solver_opt.c
 * @brief An optimized C implementation of a matrix solver using loop-level optimizations.
 *
 * This file provides a version of the matrix solver that applies several
 * micro-optimizations to the naive, loop-based implementation. It computes
 * the same mathematical expression but aims for better performance by
 * giving hints to the compiler and manually managing memory access patterns.
 */
#include "utils.h"


/**
 * @brief Performs a sequence of matrix operations using optimized naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *
 *       Optimizations applied:
 *       1.  **Register Keyword**: The `register` keyword is used as a hint to the
 *           compiler to store frequently accessed variables (loop counters and
 *           pointers) in CPU registers for faster access. Modern compilers
 *           often handle this automatically, but it shows the intent to optimize.
 *       2.  **Pointer Arithmetic**: Instead of array indexing (e.g., `A[i*N + k]`),
 *           this version uses pointers that are manually incremented within the
 *           loops. This can reduce the overhead of address calculation in tight loops.
 *       3.  **Loop Fusion**: The final multiplication and addition are fused into a
 *           single loop block for better data locality.
 */
double* my_solver(int N, double *A, double* B) {
	register int i, j, k;
	printf("OPT SOLVER
");

	// Allocate memory for intermediate and final result matrices.
	double *AB = (double*) calloc(N * N, sizeof(double));
	if (AB == NULL) {
		printf("Error calloc");
		return NULL;
	}
	// Note: ABBt is allocated but never used.
	double *ABBt = (double*) calloc(N * N, sizeof(double));
	if (ABBt == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *AtA = (double*) calloc(N * N, sizeof(double));
	if (AtA == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Error calloc");
		return NULL;
	}


	
	// Block Logic: Compute AB = A * B, where A is treated as an upper triangular matrix.
	for (i = 0; i < N; ++i) {
		// Optimization: Get a base pointer to the current row of A.
		register double *orig_pA = &A[i * N + i];

		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			register double *pA = orig_pA;
			// Optimization: Get a base pointer to the current column of B.
			register double *pB = &B[i * N + j];

			// The inner loop for k starts from i, assuming A is upper triangular.
			for (k = i; k < N; ++k) {
				sum += *pA * *pB;
				pA++;    // Move to the next element in A's row.
				pB += N; // Move to the next element in B's column.
			}
			AB[i * N + j] = sum;
		}
	}

	
	// Block Logic: Compute AtA = A^T * A.
	for (i = 0; i < N; ++i) {
		// Optimization: Get a base pointer to the current column of A (treating it as A-transpose).
		register double *orig_pAt = &A[i];

		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			register double *pAt = orig_pAt;
			register double *pA = &A[j];

			// This loop computes the dot product of column i and column j of A.
			for (k = 0; k <= i ; ++k) {
				sum += *pAt * *pA;
				pAt += N; // Move down the column of A-transpose.
				pA += N;  // Move down the column of A.
			}
			AtA[j * N + i] = sum;
		}
	}

	
	// Block Logic: Fuse the final operations to compute C = (A * B) * B^T + A^T * A.
	for (i = 0; i < N; ++i) {
		// Optimization: Get a base pointer to the current row of AB.
		register double *orig_pAB = &AB[i * N];

		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			register double *pAB = orig_pAB;
			// Optimization: Get a base pointer to the current row of B (for B-transpose).
			register double *pB = &B[j * N];

			// This inner loop computes the (i, j)-th element of (A * B) * B^T.
			for (k = 0; k < N; ++k) {
				sum += *pAB * *pB;
				pAB++;
				pB++;
			}
			// Add the corresponding element from A^T * A and store the final result.
			C[i * N + j] = sum + AtA[i * N + j];
		}
	}

	
	// Free the memory used for intermediate matrices.
	free(AB);
	free(ABBt);
	free(AtA);

	return C;	
}
