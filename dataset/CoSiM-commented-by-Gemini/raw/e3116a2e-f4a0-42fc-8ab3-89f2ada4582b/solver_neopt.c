/**
 * @file solver_neopt.c
 * @brief Unoptimized standard implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Naive multi-loop dense matrix multiplication with upper triangular conditions.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the intermediate matrix AUX and result matrix C.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	int i, j, k;
	double *C = calloc(N*N,sizeof(double));
	
	double *AUX = calloc(N*N, sizeof(double));
	
	double res;
	
    /**
     * Block Logic: Computes the intermediate matrix product $AUX = A \times B$.
     * Invariant: AUX accumulates partial row-column dot products considering A is upper triangular.
     */
	for(i = 0 ; i < N; i++){
		for(j = 0; j < N; j++){
			res = 0;
			for(k = 0; k < N; k++){
				// Inline: Leverages the upper triangular property of matrix A to skip zero values.
				if(i <= k)
					res += A[i*N + k] * B[k*N + j];
			}
			AUX[i*N + j] = res;
		}
	}
	
    /**
     * Block Logic: Computes $C = AUX \times B^T + A^T \times A$.
     * Invariant: C stores the final elements by reusing loop structures.
     */
	for(i = 0 ; i < N; i++){
		for(j = 0; j < N; j++){
			res = 0;
			for(k = 0; k < N; k++){
					res += AUX[i*N + k] * B[j*N + k];
					// Inline: Accumulates $A^T \times A$, exploiting A's upper triangular shape.
					if(k <= j)
						res += A[k*N + i] * A[k*N + j];
			}
			C[i*N + j] = res;
		}
	}

	free(AUX);
	return C;
}
