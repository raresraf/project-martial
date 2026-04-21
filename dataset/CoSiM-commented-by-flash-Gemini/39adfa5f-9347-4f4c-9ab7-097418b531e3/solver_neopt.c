
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Naive reference implementation for the matrix expression solver.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = A * (B * B^T) + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Standard nested-loop matrix arithmetic.
 * Time Complexity: $O(N^3)$ due to multiple triple-nested matrix multiplications.
 * Space Complexity: $O(N^2)$ for intermediate matrices (BBt, AtA) and the result.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

    /**
     * Block Logic: Compute BBt = B * B^T.
     * Logic: Implements multiplication by accessing B[j][k] as B^T[k][j].
     * Invariant: BBt[i][j] stores the dot product of B's i-th and j-th rows.
     */
	double *BBt = (double *)calloc(N * N, sizeof(double));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

    /**
     * Block Logic: Compute A_BBt = A * BBt.
     * Optimization: Exploit's A's upper triangularity by starting k from i.
     */
    double *A_BBt = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = i; k < N; k++) {
				A_BBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

    /**
     * Block Logic: Compute AtA = A^T * A.
     * Optimization: Exploys upper triangularity by limiting k up to i. 
     * Handles transposition by accessing A[k][i] instead of A[i][k].
     */
    double *AtA = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k <= i; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

    /**
     * Block Logic: Aggregate the two matrix products into the final result.
     */
    for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A_BBt[i * N + j] += AtA[i * N + j];
		}
	}

    free(AtA);
    free(BBt);
	return A_BBt;
}
