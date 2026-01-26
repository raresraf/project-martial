/**
 * @file solver_neopt.c
 * @brief Implements a matrix solver using naive nested loop (non-optimized) matrix operations.
 * Algorithm: Direct implementation of matrix operations using three nested loops,
 *            performing conditional accumulations based on indices.
 * Time Complexity: O(N^4) due to nested loops within nested loops for matrix operations.
 * Space Complexity: O(N^2) for two auxiliary matrices (`aux1`, `aux2`).
 */

#include "utils.h"

/**
 * @brief Solves a matrix problem using non-optimized nested loops.
 *
 * This function performs a series of matrix operations on input matrices A and B.
 * It uses auxiliary matrices `aux1` and `aux2` for intermediate calculations.
 * The exact mathematical operation is complex due to conditional accumulations
 * within the innermost loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A Pointer to the N x N matrix A.
 * @param B Pointer to the N x N matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	// i, j, k: Loop counters for iterating through matrix elements.
	int i, j, k;
	// aux1: First auxiliary N x N matrix, initialized to zeros.
	double *aux1 = (double*) calloc(N * N, sizeof(double));
	// aux2: Second auxiliary N x N matrix, initialized to zeros.
	double *aux2 = (double*) calloc(N * N, sizeof(double));

	// Block Logic: First set of complex conditional matrix accumulations.
	// This block calculates intermediate results into `aux1` and `aux2` based on
	// specific index relationships (i <= k) and (k <= i && k <= j).
	// This likely represents a partial matrix multiplication or a specialized transformation.
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
				for (k = 0; k < N; k++) {
					if (i <= k) {
						// Calculates elements of aux1 based on A and B.
						aux1[N * i + j] += A[N * i + k] * B[N * k + j];
					} 
					if (k <= i && k <= j) {  
						// Calculates elements of aux2 based on A.
						aux2[N * i + j] += A[N * k + i] * A[N * k + j];
					}
				}
		}
	}

	// Block Logic: Second matrix multiplication involving aux1 and B.
	// This block performs a matrix multiplication where `aux1` is multiplied by `B^T`
	// (due to B[N * j + k] which means B[k][j] when B is B[row][col])
	// and added to `aux2`. This completes the overall computation.
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				aux2[N * i + j] += aux1[N * i + k] * B[N * j + k];
			}
		}
	}

	free(aux1); // Free memory allocated for aux1 as it's no longer needed.
	return aux2; // Return the final result stored in aux2.
}
