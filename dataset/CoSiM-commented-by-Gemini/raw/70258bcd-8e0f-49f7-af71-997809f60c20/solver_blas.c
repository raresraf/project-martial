
/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 * @note This version has a slightly different implementation strategy compared to other BLAS files in the dataset,
 *       though it computes the same mathematical expression.
 */
#include "utils.h"
#include <string.h>
#include "cblas.h"


/**
 * @brief Performs a sequence of matrix operations using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N), treated as upper triangular in one step.
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @note The function computes the following expression: C = (A * B) * B^T + A^T * A
 *       This is achieved through a series of BLAS calls:
 *       1. `cblas_dtrmm` to compute the product of a triangular matrix A and matrix B.
 *       2. `cblas_dgemm` to compute the product of A-transpose and A.
 *       3. `cblas_dgemm` to compute the product of (A*B) and B-transpose and add the result to the previous step.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");

	// Allocate memory for intermediate and final result matrices.
	double *AB = (double*) calloc(N * N, sizeof(double));
	if (AB == NULL) {
		printf("Error calloc");
		return NULL;
	}
	// Note: ABBt is allocated but never used in this implementation.
	double *ABBt = (double*) calloc(N * N, sizeof(double));
	if (ABBt == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *AtA = (double*) calloc(N * N, sizeof(double));
	if (AtA == NULL) {
		printf("Error calloc");
		return NULL;
	}

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Error calloc");
		return NULL;
	}

	
	// Step 1: Compute AB = A * B.
	// First, copy B into AB.
	memcpy(AB, B, N * N * sizeof(double));

	// Then, perform AB = A * AB, where A is an upper triangular matrix.
	// cblas_dtrmm: Triangular Matrix-Matrix Multiply.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);

	
	// Step 2: Compute AtA = A^T * A.
	// cblas_dgemm: General Matrix-Matrix Multiply.
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1.0, A, N, A, N, 0.0, AtA, N);

	
	// Step 3: Initialize the final result matrix C with the value of AtA.
	memcpy(C, AtA, N * N * sizeof(double));

	
	// Step 4: Compute C = (A * B) * B^T + C.
	// This performs C = 1.0 * AB * B^T + 1.0 * C, where AB = A * B from Step 1.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);

	// Free the memory used for intermediate matrices.
	free(AB);
	free(ABBt); // Freeing the unused allocated memory.
	free(AtA);

	// Return the final result.
	return C;
}
