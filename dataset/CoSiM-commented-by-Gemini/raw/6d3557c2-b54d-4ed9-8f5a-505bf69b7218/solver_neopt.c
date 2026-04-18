
/**
 * @file solver_neopt.c
 * @brief A non-optimized, naive implementation of a matrix solver.
 *
 * This file provides a basic C implementation for a series of matrix operations,
 * performing the calculations using explicit for-loops. This version is intended
 * for correctness checking and as a performance baseline against optimized
 * versions (e.g., using BLAS).
 */

#include "utils.h"





/**
 * @brief Performs a sequence of matrix operations using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix F. The caller is responsible for freeing this memory.
 *
 * @note This function computes the same expression as the BLAS version: F = (A * B) * B^T + A^T * A.
 *       It uses explicit, nested for-loops for all matrix multiplications, which is
 *       significantly less performant than using an optimized BLAS library.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER
");
	int i, j, k;

	
	// Block Logic: Compute C = A * B, where A is treated as an upper triangular matrix.
	double *C = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			
			
			// Inner loop starts from k=i, assuming A is upper triangular (A[i][k] is zero for k < i).
			for (k = i; k < N; k++) {
				*(C + i * N + j) += *(A + i * N + k) * *(B + k * N + j);
			}
		}
	}

	
	// Block Logic: Compute D = C * B^T (C multiplied by the transpose of B).
	double *D = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				// Accessing B in a transposed manner: B[j][k] is used instead of B[k][j].
				*(D + i * N + j) += *(C + i * N + k) * *(B + j * N + k);
			}
		}
	}

	
	// Block Logic: Compute E = A^T * A (transpose of A multiplied by A).
	double *E = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int min = i < j ? i : j;
			
			// Accessing A in a transposed manner for the first term: A[k][i] is used.
			for (k = 0; k <= min; k++) {
				*(E + i * N + j) += *(A + k * N + i) * *(A + k * N + j);
			}
		}
	}

	// Allocate memory for the final result matrix F.
	double *F = calloc(N * N, sizeof(double));
	
	// Block Logic: Perform the final addition F = D + E.
	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
			*(F + i * N + j) = *(D + i * N + j) + *(E + i * N + j);
        }
    }

	// Free the memory used for intermediate matrices.
	free(C);
	free(D);
	free(E);
	return F;
}
