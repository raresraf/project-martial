
/**
 * @file solver_neopt.c
 * @brief A non-optimized, naive implementation of a matrix solver.
 *
 * This file provides a basic C implementation for a series of matrix operations,
 * performing the calculations using explicit for-loops. This version is intended
 * for correctness checking and as a performance baseline against optimized
 * versions. It differs from other non-optimized versions by fusing the final
 * multiplication and addition steps.
 */
#include "utils.h"


 
/**
 * @brief Performs a sequence of matrix operations using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It uses explicit, nested for-loops for all matrix multiplications.
 *       The final multiplication and addition are fused into a single loop.
 */
double* my_solver(int N, double *A, double* B) {
	int i, j, k;
	printf("NEOPT SOLVER
");

	// Allocate memory for intermediate and final result matrices.
	double *AB = (double*) calloc(N * N, sizeof(double));
	if (AB == NULL) {
		printf("Error calloc
");
		return NULL;
	}
	// Note: ABBt is allocated but never used in this implementation.
	double *ABBt = (double*) calloc(N * N, sizeof(double));
	if (ABBt == NULL) {
		printf("Error calloc
");
		return NULL;
	}

	double *AtA = (double*) calloc(N * N, sizeof(double));
	if (AtA == NULL) {
		printf("Error calloc
");
		return NULL;
	}

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Error calloc
");
		return NULL;
	}

	
	// Block Logic: Compute AB = A * B, where A is treated as an upper triangular matrix.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			// Inner loop for k starts from i, assuming A is upper triangular.
			for (k = i; k < N; ++k) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	// Block Logic: Compute AtA = A^T * A.
	// The result is stored in a transposed manner in AtA.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			// This loop computes the dot product of column i and column j of A.
			for (k = 0; k <= i; ++k) {
				AtA[j * N + i] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	// Block Logic: Fuse the final operations to compute C = (A * B) * B^T + A^T * A.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			double sum = 0;
			// This inner loop computes the (i, j)-th element of (A * B) * B^T.
			for (k = 0; k < N; ++k) {
				sum += AB[i * N + k] * B[j * N + k];
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
