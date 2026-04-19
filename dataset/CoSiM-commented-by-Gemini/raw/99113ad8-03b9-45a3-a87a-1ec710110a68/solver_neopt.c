/**
 * @raw/99113ad8-03b9-45a3-a87a-1ec710110a68/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Standard nested loops traversing the matrices.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate target buffers.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
	 */
	double *C1 = calloc(N*N, sizeof(double));
	double *C = calloc(N*N, sizeof(double));
	int i,j,k;
	
	/**
	 * Block Logic: Computes the partial matrix multiplication C1 = A * B.
	 * Invariant: Exploits the upper triangular nature of matrix A by skipping elements where i > k.
	 */
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			for(k=0; k<N; k++){
				if(i > k)
					continue;
				C1[i*N + j] += A[i*N + k] * B[k*N + j];
			}
		}
	}

	/**
	 * Block Logic: Computes C = C1 * B^T.
	 * Invariant: Replicates the transposition of matrix B conceptually by reversing matrix indices to `[j*N + k]`.
	 */
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			C[i*N + j] = 0;
			for(k=0; k<N; k++){
				C[i*N + j] += C1[i*N + k] * B[j*N + k];
			}
		}
	}

	/**
	 * Block Logic: Solves C = C + A^T * A.
	 * Invariant: Matrix A constraints handled dynamically by skipping iteration for items below diagonal (i < k).
	 */
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			for(k=0; k<N; k++){
				if(i < k)
					continue;
				C[i*N + j] += A[k*N + i] * A[k*N + j];
			}
		}
	}
	free(C1);
	return C;
}
