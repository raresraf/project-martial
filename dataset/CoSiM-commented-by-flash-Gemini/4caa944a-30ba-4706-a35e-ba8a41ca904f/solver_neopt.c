
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
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;
	double *C, *AtA, *AB, *ABBt;

	// Logic: Pre-allocation and initialization of result and workspace buffers.
	C = calloc(N * N, sizeof(double));
	DIE(C == NULL, "calloc C");

	AtA = calloc(N * N, sizeof(double));
	DIE(AtA == NULL, "calloc AtA");

	AB = calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	ABBt = calloc(N * N, sizeof(double));
	DIE(ABBt == NULL, "calloc ABBt");

    /**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits upper triangularity of A by starting k from i.
	 * Invariant: AB[i][j] stores the dot product of A's i-th row and B's j-th column.
	 */
	for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = i; k < N; k++) {
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Logic: Computes product with B^T by accessing B[j][k] (equivalent to B^T[k][j]).
	 */
	for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
            }
        }
    }

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Exploits upper triangularity of A by limiting k up to min(i, j). 
	 * Logic: Implements transposition by accessing A[k][i] instead of A[i][k].
	 */
	for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k <= i && k <= j; k++) {
                AtA[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }

	/**
	 * Block Logic: Final accumulation Result = ABBt + AtA.
	 */
	for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
        }
    }

	free(AB);
    free(AtA);
    free(ABBt);

	return C;
}
