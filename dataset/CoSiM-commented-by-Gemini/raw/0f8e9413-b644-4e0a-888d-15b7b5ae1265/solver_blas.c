/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using a BLAS library (CBLAS interface).
 *
 * This file provides a `my_solver` implementation that leverages BLAS functions
 * to compute the expression `C = (A * B) * B^T + A^T * A`. It uses two auxiliary
 * matrices to store intermediate results during the calculation.
 */
#include "utils.h"
#include <cblas.h>

/**
 * @brief Solves a matrix equation using a sequence of BLAS function calls.
 *
 * This function computes the expression `C = (A * B) * B^T + A^T * A` for given
 * N x N matrices A and B. It uses `cblas_dtrmm` for triangular matrix
 * multiplications and `cblas_dgemm` for the final general matrix multiplication
 * and addition.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A, assumed to be upper triangular.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The implementation performs the following steps:
 * 1. Allocates two temporary matrices, `C` and `aux`.
 * 2. Copies `B` to `C` and `A` to `aux`.
 * 3. Computes `aux = A^T * A` using `cblas_dtrmm`.
 * 4. Computes `C = A * B` using `cblas_dtrmm`.
 * 5. Computes the final result `aux = (C * B^T) + aux` using `cblas_dgemm`.
 * 6. Frees the temporary matrix `C` and returns `aux`.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");
	double *C = malloc(sizeof(double) * N * N);
	double *aux = malloc(sizeof(double) * N * N);
	for (int i = 0; i < N*N; ++i) {
		C[i] = B[i];
		aux[i] = A[i];
	}
	
	
	cblas_dtrmm(	CblasRowMajor,
			CblasLeft,
			CblasUpper,
			CblasTrans,
			CblasNonUnit,
			N,
			N,
			1,
			A,
			N,
			aux,
			N
		);
	
	cblas_dtrmm(	CblasRowMajor,
			CblasLeft,
			CblasUpper,
			CblasNoTrans,
			CblasNonUnit,
			N,
			N,
			1,
			A,
			N,
			C,
			N
		);
	
	
	cblas_dgemm(	CblasRowMajor,
			CblasNoTrans,
			CblasTrans,
			N,
			N,
			N,
			1,
			C,
			N,
			B,
			N,
			1,
			aux,
			N
		);

	free(C);
	return aux;
}
