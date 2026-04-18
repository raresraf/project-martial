
/**
 * @file solver_neopt.c
 * @brief A non-optimized, loop-based implementation of a matrix solver.
 *
 * This file provides a basic C implementation for a series of matrix operations.
 * It uses a single auxiliary buffer to store different intermediate results,
 * making it more memory-efficient than versions that allocate a new buffer
 * for every intermediate step.
 */
#include "utils.h"

/**
 * @brief Performs a sequence of matrix operations using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It reuses a single auxiliary buffer `aux` for multiple steps.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER
");
	double *C = malloc(N * N * sizeof(*C));
	double *aux = malloc(N * N * sizeof(*aux));
	int i, j, k;

	
	// Block Logic: Step 1 - Compute aux = A * B, treating A as an upper triangular matrix.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			// Zero out the element before accumulating.
			aux[i * N + j] = 0;
			// Inner loop for k starts from i, assuming A is upper triangular.
			for (k = i; k < N; ++k) {
				aux[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	// Block Logic: Step 2 - Compute C = aux * B^T, which is (A * B) * B^T.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = 0;
			for (k = 0; k < N; ++k) {
				// Accessing B in a transposed manner: B[j*N + k] is used instead of B[k*N + j].
				C[i * N + j] += aux[i * N + k] * B[j * N + k];
			}
		}
	}

	
	// Block Logic: Step 3 - Overwrite aux to compute aux = A^T * A.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			aux[i * N + j] = 0;
			// This computes the dot product of column i and column j of A.
			for (k = 0; k < i + 1; ++k) {
				aux[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	// Block Logic: Step 4 - Perform the final addition C = C + aux.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	// Free the auxiliary buffer and return the result.
	free(aux);
	return C;
}
