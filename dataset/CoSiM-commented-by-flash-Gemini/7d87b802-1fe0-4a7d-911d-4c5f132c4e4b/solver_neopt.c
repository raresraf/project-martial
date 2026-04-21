/**
 * @7d87b802-1fe0-4a7d-911d-4c5f132c4e4b/solver_neopt.c
 * @brief Naive implementation of a matrix expression solver.
 *
 * Functional Utility: Implements the matrix expression Result = (A * B) * B^T + (A^T * A)
 * using standard triple-nested iterative loops. Serves as a correctness reference 
 * and performance baseline.
 *
 * Domain: HPC Numerical Baselines.
 */

#include "utils.h"


/**
 * @brief Computes the matrix expression via sequence of naive stages.
 * Algorithm: Stage-based linear algebra calculation.
 * Time Complexity: $O(N^3)$.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	// Pre-condition: Allocates stack-based VLAs for intermediate product buffers.
	double c[N][N], d[N][N];
	double *e = malloc(N * N * sizeof(double));
	
	/**
	 * Block Logic: Compute stage 1: C = A * B.
	 * Invariant: Exploits the upper triangular property of A by using a conditional 
	 * check (i <= k) within the inner loop.
	 */
	for(int i = 0; i < N; 	i++) {
		for(int j = 0; j < N; j++) {
				c[i][j] = 0.0;
				for (int k = 0; k < N; k++) {
					if(i <= k) {
						c[i][j] += A[i * N + k] * B[k * N + j];
					}
					
				}
		}
	}

	/**
	 * Block Logic: Compute stage 2: D = C * B^T.
	 * Logic: Implicitly transposes matrix B by accessing its elements as B[j * N + k].
	 */
	for(int i = 0; i < N; 	i++) {
		for(int j = 0; j < N; j++) {
			d[i][j] = 0.0;
			for (int k = 0; k < N; k++) {
				d[i][j] += c[i][k] * B[j * N + k];
			}
		}
	}

	
	/**
	 * Block Logic: Compute stage 3: Result = (A^T * A) + D.
	 * Invariant: Aggregates the Gramian term (A^T * A) and the previously computed term D.
	 * Logic: The conditional (k <= i || k < j) attempts to optimize for A's triangularity.
	 */
	for(int i = 0; i < N; 	i++) {
		for(int j = 0; j < N; j++) {
			e[i * N + j]= 0.0;

			for (int k = 0; k < N; k++) {
				if(k <= i || k < j) {
					e[i * N + j] += A[k * N + i] *  A[k * N + j];
				}
				
			}

			e[i * N + j] += d[i][j];
		}
	}

	
	return e;

}
