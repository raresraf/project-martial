
/**
 * @file solver_neopt.c
 * @brief A non-optimized, loop-based implementation of a matrix solver.
 *
 * This file provides a basic C implementation for a series of matrix operations,
 * calculating the result in a clear, step-by-step fashion using intermediate
 * matrices for each stage of the computation.
 */
#include "utils.h"

/**
 * @brief Performs a sequence of matrix operations using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It uses explicit, nested for-loops for all matrix multiplications and
 *       stores intermediate results in separate buffers.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER
");
	int i, j, k;
	// Allocate memory for intermediate matrices.
	double *M1 = (double *)calloc(N * N, sizeof(double)); 
	double *M2 = (double *)calloc(N * N, sizeof(double)); 
	double *M3 = (double *)calloc(N * N, sizeof(double)); 
	if (!M1 || !M2 || !M3)
		return NULL;

	// Block Logic: Step 1 - Compute M1 = A * B, treating A as an upper triangular matrix.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			// Inner loop for k starts from i, assuming A is upper triangular.
			for(k = i; k < N; k++)
				M1[i * N + j] += A[i * N + k] * B[k * N + j];

	// Block Logic: Step 2 - Compute M2 = M1 * B^T, which is (A * B) * B^T.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for(k = 0; k < N; k++)
			 	// Accessing B in a transposed manner: B[j*N + k] is used instead of B[k*N + j].
			 	M2[i * N + j] += M1[i * N + k] * B[j * N + k];

	// Block Logic: Step 3 - Compute M3 = A^T * A.
	int end;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			if (i < j)
				end = i;
			else
				end = j;
			// This computes the dot product of column i and column j of A.
			for(k = 0; k <= end; k++)
			 	M3[i * N + j] += A[k * N + i] * A[k * N + j];
			}

	// Block Logic: Step 4 - Perform the final addition M2 = M2 + M3.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			M2[i * N + j] += M3[i * N + j];

	// Free intermediate matrices. The result is in M2.
	free(M1);
	free(M3);
	return M2;
}
