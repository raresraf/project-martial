
/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 * @note This version cleverly reuses allocated matrices to store different
 *       intermediate results throughout the calculation.
 */
#include "utils.h"
#include<cblas.h>
#include<string.h>

/**
 * @brief Performs a sequence of matrix operations using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * @note The function computes the expression: C = A * (B * B^T) + A^T * A.
 *       It follows these steps:
 *       1. Computes M1 = B * B^T.
 *       2. Computes M2 = A^T * A (by first copying A to M2, then performing the operation in-place).
 *       3. Computes the final result M2 = A * M1 + M2.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");
	// Allocate memory for intermediate matrices.
	double *M2 = (double *) malloc(N * N * sizeof(double));
	double *M1 = (double *) calloc(N * N, sizeof(double));
	// Initialize M2 with the contents of A.
	memcpy(M2, A, N * N * sizeof(double));

	
 	// Step 1: Compute M1 = B * B^T.
	// cblas_dgemm: General Matrix-Matrix Multiply. Result is stored in M1.
 	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 1.0, M1, N);
	
	// Step 2: Compute M2 = A^T * A.
	// This operation is done in-place on M2, which was initialized with A.
	// cblas_dtrmm: Triangular Matrix-Matrix Multiply.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, M2, N);
	
	// Step 3: Compute M2 = A * M1 + M2.
	// This calculates A * (B * B^T) and adds the previously computed A^T * A.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, M1, N, 1.0, M2, N);

	// Free the M1 intermediate matrix. M2 holds the final result.
	free(M1);
	return M2;
}
