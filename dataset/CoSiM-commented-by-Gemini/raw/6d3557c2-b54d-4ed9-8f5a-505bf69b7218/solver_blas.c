
/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 *
 * This file contains a function that performs a series of matrix operations
 * on two input matrices, A and B, leveraging a CBLAS interface for
 * high-performance linear algebra computations.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>


/**
 * @brief Performs a sequence of matrix operations using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix D. The caller is responsible for freeing this memory.
 *
 * @note The function computes the following expression: D = (A * B) * B^T + A^T * A
 *       where A is treated as an upper triangular matrix in the first operation.
 *       It uses cblas_dtrmm for triangular matrix multiplication and cblas_dgemm
 *       for general matrix-matrix multiplication.
 */
double* my_solver(int N, double *A, double *B) {
	
	int i, j;

	
	// Allocate memory for matrix C and initialize it with the values of B.
	double *C = calloc(N * N, sizeof(double));
	memcpy(C, B, N * N * sizeof(double));

	
	// Perform C = A * C, where A is an upper triangular matrix.
	// cblas_dtrmm: Triangular Matrix-Matrix Multiply.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
    

	// Allocate memory for matrix D.
	double *D = calloc(N * N, sizeof(double));
	
	// Perform D = 1.0 * C * B^T + 0 * D.
	// cblas_dgemm: General Matrix-Matrix Multiply.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, C, N, B, N, 0, D, N);

	// Allocate memory for matrix E and initialize it with the values of A.
	double *E = calloc(N * N, sizeof(double));
	memcpy(E, A, N * N * sizeof(double));
	
	// Perform E = A^T * E, where A is an upper triangular matrix.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, E, N, E, N);

	
	// Block Logic: Perform the final addition D = D + E.
	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
			*(D + i * N + j) = *(D + i * N + j) + *(E + i * N + j);
        }
    }

	// Free the memory used for intermediate matrices C and E.
	free(C);
	free(E);
	return D;
}
