/**
 * @file solver_blas.c
 * @brief A BLAS-based, high-performance implementation of a matrix expression solver.
 *
 * This file implements the matrix expression calculation using functions from a
 * Basic Linear Algebra Subprograms (BLAS) library through its C interface (CBLAS).
 * This approach offloads the computationally intensive matrix multiplications
 * to a highly optimized, platform-specific library, resulting in significant
 * performance gains compared to a naive implementation.
 *
 * The expression calculated is (A * B * B^T) + (A^T * A).
 */

#include "utils.h"
#include <cblas.h>

/**
 * @brief Copies the contents of one matrix to another.
 * @param N The dimension of the matrices.
 * @param copyM A pointer to the destination matrix.
 * @param M A pointer to the source matrix.
 */
void copy(int N, double **copyM, double *M) {
	for(int i = 0; i < N*N; i++) {
		(*copyM)[i] = M[i];
	}
}


/**
 * @brief Calculates the matrix expression (A * B * B^T) + (A^T * A) using BLAS functions.
 * @param N The dimension of the square matrices A and B.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * Algorithm: This function leverages BLAS routines to efficiently compute the expression.
 * It follows these steps:
 * 1.  `AB = A * B` is computed using `cblas_dtrmm`, which performs a triangular
 *     matrix-matrix multiplication.
 * 2.  `AtA = A^T * A` is also computed using `cblas_dtrmm` with the transpose flag.
 * 3.  The final result `res = (AB * B^T) + AtA` is computed in a single call to
 *     `cblas_dgemm` (general matrix-matrix multiply), which performs the operation
 *     `C = alpha*A*B + beta*C`.
 */
double* my_solver(int N, double *A, double *B) {
	// Step 1: Compute AB = A * B.
	// We use cblas_dtrmm (triangular matrix-matrix multiply).
	// The operation is B := A * B, where A is upper triangular.
	double *AB = malloc((N * N) * sizeof(double));
	copy(N, &AB, B); // AB is initialized with B.
	cblas_dtrmm(CblasRowMajor,     // Matrix layout
				CblasLeft,         // A is on the left: A * B
				CblasUpper,        // A is an upper triangular matrix
				CblasNoTrans,      // Use A, not its transpose
				CblasNonUnit,      // A is not assumed to have a unit diagonal
				N, N, 1.0,         // Dims and alpha scalar (1.0)
				A, N,              // Matrix A and its leading dimension
				AB, N);            // Matrix B (input and output) and its leading dim

	// Step 2: Compute AtA = A^T * A.
	// We use cblas_dtrmm again.
	// The operation is A := A^T * A, where A from A^T is upper triangular.
	double *AtA = malloc((N * N) * sizeof(double));
	copy(N, &AtA, A); // AtA is initialized with A.
	cblas_dtrmm(CblasRowMajor,     // Matrix layout
				CblasLeft,         // A^T is on the left: A^T * A
				CblasUpper,        // The original matrix A is upper triangular
				CblasTrans,        // Use the transpose of A
				CblasNonUnit,      // A is not unit triangular
				N, N, 1.0,         // Dims and alpha
				A, N,              // Matrix A (to be transposed)
				AtA, N);           // Matrix A (input and output)

	// Step 3: Compute the final result: res = (AB * B^T) + AtA
	// We use cblas_dgemm (general matrix-matrix multiply) which can compute
	// C = alpha*A*B + beta*C.
	// Here, we set A=AB, B=B^T, C=AtA, alpha=1.0, beta=1.0.
	double *res = malloc((N * N) * sizeof(double));
	copy(N, &res, AtA); // res is initialized with AtA.
	cblas_dgemm(CblasRowMajor,     // Matrix layout
				CblasNoTrans,      // Don't transpose the first matrix (AB)
				CblasTrans,        // Transpose the second matrix (B)
				N, N, N,           // Dims of matrices
				1.0,               // alpha = 1.0
				AB, N,             // First matrix AB
				B, N,              // Second matrix B (to be transposed)
				1.0,               // beta = 1.0
				res, N             // Result matrix C (initialized with AtA)
				);

	// Free intermediate allocated memory
	free(AB);
	free(AtA);
	return res;
}
