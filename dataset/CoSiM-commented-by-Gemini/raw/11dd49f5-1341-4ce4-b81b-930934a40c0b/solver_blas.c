/**
 * @file solver_blas.c
 * @brief BLAS-based implementation of a matrix equation solver.
 * @details This file contains a function that computes the result of the
 * matrix equation: result = (A * B) * B^T + A^T * A, where A and B are
 * N x N matrices. It leverages the CBLAS interface for high-performance
 * Basic Linear Algebra Subprograms.
 */
#include "utils.h"
#include "cblas.h"
#include <string.h>

/**
 * @brief Computes the matrix expression (A * B) * B^T + A^T * A.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix. The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double *B) {
	// Allocate memory for intermediate and final result matrices.
	// The 'register' keyword is a hint to the compiler for optimization,
	// though modern compilers often manage registers more effectively.
	register double *ab = (double *) malloc(N * N * sizeof(double));
	register double *ata = (double *) malloc(N * N * sizeof(double));
	register double *abbt = (double *) malloc(N * N * sizeof(double));
	register double *result = (double *) malloc(N * N * sizeof(double));

	
	// Copy matrix A to ata to prepare for the first operation.
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			ata[i * N + j] = A[i * N + j];
		}
	}
	/**
	 * Block Logic: Compute A^T * A using triangular matrix multiplication.
	 * cblas_dtrmm computes ata = alpha * op(A) * B, where op(A) is A or A^T.
	 * Here, it computes ata = 1.0 * A^T * ata (where ata is initially A).
	 * This effectively calculates A^T * A, assuming A is upper triangular.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
				CblasNonUnit, N, N, 1, A, N, ata, N);

	
	// Copy matrix B to ab to prepare for the second operation.
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			ab[i * N + j] = B[i * N + j];		
		}
	}
	/**
	 * Block Logic: Compute A * B using triangular matrix multiplication.
	 * Here, it computes ab = 1.0 * A * ab (where ab is initially B).
	 * This calculates A * B, assuming A is upper triangular.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N, 1, A, N, ab, N);
	
	
	/**
	 * Block Logic: Compute (A * B) * B^T using general matrix-matrix multiplication.
	 * cblas_dgemm computes C = alpha * op(A) * op(B) + beta * C.
	 * Here, it computes abbt = 1.0 * ab * B^T + 0.0 * abbt,
	 * resulting in abbt = (A * B) * B^T.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, ab, N, B, N, 0, abbt, N);


	
	/**
	 * Block Logic: Sum the intermediate results to get the final matrix.
	 * result = abbt + ata => (A * B) * B^T + A^T * A
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			result[i * N + j] = abbt[i * N + j] + ata[i * N + j];
		}
	}


	// Free the memory allocated for intermediate matrices.
	free(ab);
	free(abbt);
	free(ata);

	return result;
}
