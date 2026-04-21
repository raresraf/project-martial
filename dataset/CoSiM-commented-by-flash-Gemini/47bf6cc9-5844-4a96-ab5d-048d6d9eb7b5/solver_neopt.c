
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
 * Space Complexity: $O(N^2)$ for multiple intermediate workspace matrices (AB, ABBt, AtA).
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *ABBt;
	double *AtA;
	double *C;
	int i, j, k;

	// Logic: Pre-allocation and zero-initialization of workspaces.
	AB = calloc(N * N,  sizeof(double));
	ABBt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	AtA = calloc(N * N, sizeof(double));

	// Guard: Ensure memory was successfully provisioned.
	if (AB == NULL || ABBt == NULL || C == NULL || AtA == NULL) {
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits upper triangularity of A by starting k from i.
	 */
	for (i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
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
		for(j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
               	ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
            }
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Sparse dot-product logic.
	 * Logic: Accesses columns of A to compute the Gramian matrix. 
	 * The 'break' conditional terminates the dot-product early once a zero 
	 * element is encountered, further exploiting A's triangularity.
	 */
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				if (A[N * k + i] == 0 || A[N * k + j] == 0)
					break;
				AtA[N * i + j] += A[N * k + i] * A[N * k + j];
			}
		}
	}

	/**
	 * Block Logic: Final summation pass.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
