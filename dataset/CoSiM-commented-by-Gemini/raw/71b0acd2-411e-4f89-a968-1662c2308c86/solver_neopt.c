
/**
 * @file solver_neopt.c
 * @brief A non-optimized, loop-based implementation of a matrix solver.
 *
 * This file provides a basic C implementation for a series of matrix operations.
 * It uses explicit for-loops for all calculations but includes an optimization
 * to exploit the symmetry of intermediate matrix products.
 */
#include "utils.h"


/**
 * @brief Allocates memory for several intermediate and final matrices.
 * @param N The dimension of the matrices.
 * @param C Pointer to the final result matrix.
 * @param BA_t Pointer to an intermediate matrix (used for B*B^T).
 * @param AA_t Pointer to an intermediate matrix (used for A^T*A).
 * @param AAB Pointer to an intermediate matrix (used for A*(B*B^T)).
 * @note The parameter names `BA_t` and `AAB` are somewhat misleading given their usage.
 */
void allocate_matrices(int N, double **C, double **BA_t, double **AA_t,
	double **AAB)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*BA_t = calloc(N * N, sizeof(**BA_t));
	if (NULL == *BA_t)
		exit(EXIT_FAILURE);

	*AA_t = calloc(N * N, sizeof(**AA_t));
	if (NULL == *AA_t)
		exit(EXIT_FAILURE);

	*AAB = calloc(N * N, sizeof(**AAB));
	if (NULL == *AAB)
		exit(EXIT_FAILURE);
}

/**
 * @brief Performs a sequence of matrix operations using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C.
 *
 * @note This function computes the expression: C = A * (B * B^T) + A^T * A.
 *       It uses manual loops but optimizes the calculation of the symmetric
 *       matrices B*B^T and A^T*A.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER
");

	double *C;
	double *BB_t;
	double *AA_t;
	double *ABB_t;

	int i, j, k;
	
	// Allocate memory for all matrices.
	allocate_matrices(N, &C, &BB_t, &AA_t, &ABB_t);

	
	// Block Logic: Compute BB_t = B * B^T.
	// Optimization: Since the result is symmetric, only the lower triangle (and diagonal)
	// is computed, and the upper triangle is filled by copying.
	for (i = 0; i < N; i++) {
		for (j = 0; j <= i; j++) {
			BB_t[i * N + j] = 0.0;
			// Computes the dot product of row i and row j of B.
			for (k = 0; k < N; k++) {
				BB_t[i * N + j] += B[i * N + k] * 
								B[j * N + k];
			}
			// Exploit symmetry: BB_t[j][i] = BB_t[i][j].
			if(i != j) {
				BB_t[j * N + i] = BB_t[i * N + j];
			}
		}
	}

	
	// Block Logic: Compute AA_t = A^T * A.
	// Optimization: This also exploits the symmetry of the resulting matrix.
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			AA_t[i * N + j] = 0.0;
			// Computes the dot product of column i and column j of A.
			for (k = 0; k <= i; k++) {
				AA_t[i * N + j] += A[k * N + i] * 
								A[k * N + j];
			}
			// Exploit symmetry: AA_t[j][i] = AA_t[i][j].
			if(i != j) {
				AA_t[j * N + i] = AA_t[i * N + j];
			}
		}
	}
	
	// Block Logic: Compute ABB_t = A * BB_t, where A is treated as upper triangular.
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			ABB_t[(i * N + j)] = 0.0;
			// Inner loop for k starts from i, assuming A is upper triangular.
			for (k = i; k < N; k++) {
				ABB_t[(i * N + j)] += A[i * N + k] * 
								BB_t[k * N + j];
			}
		}
	}
	
	// Block Logic: Perform the final addition C = ABB_t + AA_t.
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = ABB_t[i * N + j] + AA_t[i * N + j];

	
	// Free the memory used for intermediate matrices.
	free(BB_t);
	free(AA_t);
	free(ABB_t);

	return C;
}
