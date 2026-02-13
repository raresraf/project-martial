/**
 * @file solver_blas.c
 * @brief This file implements a matrix solver using the BLAS (Basic Linear Algebra Subprograms) library.
 * Specifically, it performs matrix operations for solving a linear system or matrix equation,
 * leveraging highly optimized routines provided by BLAS for performance.
 *
 * Algorithm: Matrix multiplication and triangular matrix multiplication using BLAS functions.
 * Time Complexity: Dominated by matrix multiplications, typically O(N^3) for N x N matrices.
 * Space Complexity: O(N^2) for storing intermediate matrices.
 */

#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "cblas.h"

/**
 * @brief Solves a matrix equation using BLAS (Basic Linear Algebra Subprograms) routines.
 * This function performs a sequence of matrix operations, including copying, triangular matrix
 * multiplication (`dtrmm`), and general matrix multiplication (`dgemm`), to compute a result matrix `C`.
 * The specific mathematical operation being solved depends on the combination of `dtrmm` and `dgemm` calls.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the input matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the newly allocated result matrix C (N x N). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	// Functional Utility: Allocate memory for result matrix C and intermediate matrix AB.
	double *C = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));

	// Functional Utility: Copy matrix B into AB.
	memcpy(AB, B, N * N * sizeof(double));
	
	// Functional Utility: Copy matrix A into C.
	memcpy(C, A, N * N * sizeof(double));	

	// Block Logic: Perform triangular matrix multiplication: C = A * C (where A is upper triangular and transposed).
	// Precondition: A is an N x N matrix, C is an N x N matrix, N is the dimension.
	// Invariant: C contains the result of A (transposed, upper triangular) multiplied by the initial C.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
	
	// Block Logic: Perform triangular matrix multiplication: AB = A * AB (where A is upper triangular).
	// Precondition: A is an N x N matrix, AB is an N x N matrix, N is the dimension.
	// Invariant: AB contains the result of A (upper triangular) multiplied by the initial AB (which was B).
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);
	
	// Block Logic: Perform general matrix multiplication: C = AB * B (transposed) + C.
	// Precondition: AB, B, C are N x N matrices, N is the dimension.
	// Invariant: C contains the final computed result.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);

	// Functional Utility: Free dynamically allocated memory for intermediate matrix AB.
	free(AB);
	
	return C;
}
