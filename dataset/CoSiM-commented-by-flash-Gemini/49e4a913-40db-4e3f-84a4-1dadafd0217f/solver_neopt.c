
#include "utils.h"
#include <stdlib.h>

/**
 * @file solver_neopt.c
 * @brief Naive matrix solver implementation with manual property management.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + (A * B) * B^T
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of standard triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate workspace matrices (AB, ABB_t) and result C.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	double * C, *AB, *ABB_t;
	int i, j, k;

	// Logic: Pre-allocation and zero-initialization of result and temporary matrices.
	C = calloc(N * N, sizeof(double));
	if(C == NULL) {
		printf("Probleme la alocarea memoriei\n");
	}
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL)
		printf("Probleme la alocarea memoriei\n");
	ABB_t = calloc(N * N, sizeof(double));
	if(ABB_t == NULL)
		printf("Probleme la alocarea memoriei\n");

	/**
	 * Block Logic: Compute C = A^T * A.
	 * Optimization: Exploits upper triangularity of A by limiting k-range (k <= i).
	 * Logic: Handles transposition by accessing column indices k*N+i.
	 * Invariant: C[i][j] stores the dot product of A's i-th and j-th columns.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k <= i; k++) {
				C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Compute ABB_t = AB * B^T.
	 * Logic: Computes product with B^T by accessing B[j][k] (equivalent to B^T[k][j]).
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Final aggregation pass.
	 * Invariant: Sums the symmetric component (AtA) and the general product component (ABBt).
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			C[i * N + j] += ABB_t[i * N + j];
		}
	}

	free(AB);
	free(ABB_t);
	return C;
}
