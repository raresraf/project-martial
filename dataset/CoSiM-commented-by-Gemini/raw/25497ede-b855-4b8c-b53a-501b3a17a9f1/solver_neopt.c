/**
 * @file solver_neopt.c
 * @brief A non-optimized, naive C implementation for solving a matrix equation.
 * @details This file provides a basic implementation of the equation
 * C = (A * B) * B' + (A' * A). It uses explicit, triply-nested loops for
 * matrix multiplication, serving as a baseline for performance comparison against
 * optimized versions like BLAS or other hand-tuned implementations. It assumes
 * that the input matrix A is upper triangular.
 *
 * @algorithm Naive, loop-based matrix multiplication.
 * @time_complexity O(N^3), resulting from the triply-nested loops.
 * @space_complexity O(N^2) for storing intermediate and final result matrices.
 */
#include "utils.h"

/**
 * @brief Solves the matrix equation C = (A * B) * B' + (A' * A) using nested loops.
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the matrix A, stored in row-major order. Assumed to be upper triangular.
 * @param B A pointer to the matrix B, stored in row-major order.
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	double *C, *A_tA, *AB;
	int i, j, k, limit;
	
	// Allocate memory for the final result C and intermediate matrices.
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		return NULL;
	
	A_tA = calloc(N * N, sizeof(double));
	if (A_tA == NULL)
		return NULL;
	
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		return NULL;
	
	// Block 1: Compute A_tA = A' * A.
	// The loops are optimized by assuming A is upper triangular.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			// Optimization: The inner loop only needs to go up to min(i, j)
			// because for k > i, A[k][i] is 0, and for k > j, A[k][j] is 0.
			if (i <= j)
				limit = i;
			else limit = j;
			for (k = 0; k <= limit; k++) 
				// A[k*N+i] accesses A column-wise, effectively transposing it.
				A_tA[i * N + j] += A[k * N + i] * A[k * N  + j];
		}
	
	// Block 2: Compute AB = A * B.
	// The loops are optimized by assuming A is upper triangular.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			// Optimization: The inner loop starts from k=i because for k < i,
			// A[i][k] would be 0 in an upper triangular matrix.
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N  + j];
		}
	
	// Block 3: Compute C = AB * B'
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				// B[j*N+k] accesses B with transposed indices.
				C[i * N + j] += AB[i * N + k] * B[j * N + k];
	
	// Block 4: Add the intermediate results to get the final C = (A*B*B') + (A'*A)
	for (i = 0; i < N; i++)
		for (j = 0;  j < N; j++)
			C[i * N + j] += A_tA[i * N + j];
			
	// Free intermediate memory.
	free(A_tA);
	free(AB);
	
	return C;
}