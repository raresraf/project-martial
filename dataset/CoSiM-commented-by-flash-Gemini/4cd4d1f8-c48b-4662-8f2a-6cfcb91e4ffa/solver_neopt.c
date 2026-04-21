
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Unoptimized (naive) implementation of the matrix expression solver.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of standard triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for several intermediate matrices (AB, ABBt, AtA).
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B)
{
	double *AtA, *C, *ABBt, *AB;

	// Logic: Workspace allocation and zero-initialization of result and temporary buffers.
	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(1);

	AtA = malloc(N * N * sizeof(*AtA));
	if (NULL == AtA)
		exit(1);

	AB = malloc(N * N * sizeof(*AB));
	if (NULL == AB)
		exit(1);

	ABBt = malloc(N * N * sizeof(*ABBt));
	if (NULL == ABBt)
		exit(1);
	
	// Pre-condition: Ensuring all accumulators start at zero.
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			AB[i * N + j] = 0;
			ABBt[i * N + j] = 0;
			AtA[i * N + j] = 0;
			C[i * N + j] = 0;
		}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits upper triangularity of A by starting k from i.
	 * Invariant: AB[i][j] stores the dot product of A's i-th row and B's j-th column.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AB[i * N + j] = 0;
			for (int k = i; k < N; k++){
				AB[i * N + j] += A[i * N + k]* B[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Logic: Computes the product with B^T by accessing B[j][k] (equivalent to B^T[k][j]).
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			ABBt[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Exploits upper triangularity of A by limiting k up to j. 
	 * Logic: Implements transposition by accessing A[k][i] instead of A[i][k].
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AtA[i * N + j] = 0;
			for (int k = 0; k <= j; k++){
				AtA[i * N + j] += A[k * N + i]* A[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Final summation pass.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
