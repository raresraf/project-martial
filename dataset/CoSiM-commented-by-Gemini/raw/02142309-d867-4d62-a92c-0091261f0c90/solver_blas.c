
/**
 * @file solver_blas.c
 * @brief Implements a matrix multiplication solver utilizing the CBLAS library.
 *
 * This module defines a solver function that performs a specific matrix
 * computation (C = D + E, where D and E involve multiplications with A and B)
 * using optimized BLAS routines for performance.
 *
 * Algorithm: Optimized matrix multiplication using CBLAS functions (cblas_dtrmm, cblas_dgemm).
 * Time Complexity: O(N^3) due to matrix multiplications, where N is the matrix dimension.
 * Space Complexity: O(N^2) for intermediate matrix storage.
 */
#include <string.h>
#include <stdlib.h>

#include "utils.h"
#include "cblas.h"


/**
 * @brief Solves a matrix computation problem using CBLAS routines.
 * @param N The dimension of the square matrices.
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C (N x N), or NULL if memory allocation fails.
 *
 * This function calculates C = (A * B^T) * B + A * A^T, where * denotes matrix
 * multiplication and ^T denotes transpose. It leverages CBLAS for optimized
 * matrix operations.
 */
double* my_solver(int N, double *A, double *B) {
	// Functional Utility: Indicates the use of the BLAS solver.
	printf("BLAS SOLVER\n");
	// Block Logic: Allocates memory for intermediate and final result matrices.
	// Invariant: All allocated pointers must be checked for NULL to prevent dereferencing issues.
	double *C = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double)); 
	double *D = calloc(N * N, sizeof(double)); 
	double *E = calloc(N * N, sizeof(double)); 

	int i;

	// Block Logic: Error handling for memory allocation failures.
	// Pre-condition: Memory allocation attempts have just been made.
	// Invariant: If any allocation fails, print an error and return NULL.
	if(C == NULL || AB == NULL || D == NULL || E == NULL) {
		printf("Eroare la alocare\n");
		return NULL;
	}

	
	// Functional Utility: Copies matrix B into AB to preserve B for later operations.
	// This is necessary because cblas_dtrmm might overwrite one of its input operands.
	memcpy(AB, B, N * N * sizeof(double));

	// Block Logic: Performs the operation AB = A * B (triangular matrix multiplication).
	// cblas_dtrmm computes A * B where A is an upper triangular matrix.
	// N: size of matrices, 1.0: scalar alpha, A: matrix A, AB: matrix B (output overwrites B)
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, AB, N);

	
	// Block Logic: Performs D = AB * B^T (general matrix multiplication).
	// cblas_dgemm computes C = alpha * A * B + beta * C.
	// Here, alpha = 1.0, beta = 0.0, A = AB, B = B^T, C = D (output).
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
		1.0, AB, N, B, N, 0.0, D, N);

	
	// Functional Utility: Copies matrix A into E to preserve A for the next operation.
	// This is necessary because cblas_dtrmm might overwrite one of its input operands.
	memcpy(E, A, N * N * sizeof(double));

	// Block Logic: Performs E = A * A^T (triangular matrix multiplication with transpose).
	// cblas_dtrmm computes A * E where A is an upper triangular matrix and E is A^T.
	// N: size of matrices, 1.0: scalar alpha, A: matrix A, E: matrix A (output overwrites E)
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, E, N);

	
	/**
	 * Block Logic: Computes the final result C by element-wise summation of D and E.
	 * Pre-condition: Matrices D and E contain the results of previous multiplications.
	 * Invariant: Each element of C is the sum of the corresponding elements in D and E.
	 */
	for(i = 0; i < N * N; i++) {
		C[i] = D[i] + E[i];
	}

	// Functional Utility: Frees memory allocated for intermediate matrices.
	free(AB);
	free(D);
	free(E);

	// Functional Utility: Returns the pointer to the result matrix C.
	return C;
}
