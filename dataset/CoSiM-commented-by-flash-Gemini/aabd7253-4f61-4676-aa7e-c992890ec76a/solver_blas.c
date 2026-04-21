
/**
 * @file solver_blas.c
 * @brief High-performance matrix solver using BLAS (Basic Linear Algebra Subprograms).
 * 
 * Functional Intent: Implements a matrix computation engine that evaluates the 
 * expression $Res = (A \times B) \times B^T + (A^T \times A)$, where $A$ is an 
 * upper triangular matrix. It leverages highly optimized BLAS routines 
 * (`cblas_dtrmm` and `cblas_dgemm`) to achieve near-peak performance for 
 * dense linear algebra operations.
 * 
 * Domain: HPC, Performance Optimization, Numerical Methods.
 */

#include "utils.h"
#include <cblas.h>

/**
 * copy - Performs a linear copy of a matrix from one memory buffer to another.
 * 
 * Time Complexity: $O(N^2)$ where N is the matrix dimension.
 */
void copy(int N, double **copyM, double *M) {
	// Block Logic: Flat-memory sweep for data replication.
	for(int i = 0; i < N*N; i++) {
		(*copyM)[i] = M[i];
	}
}


/**
 * my_solver - Orchestrates the matrix expression evaluation.
 * 
 * Algorithm: Composite BLAS orchestration.
 * 1. Computes $AB = A \times B$ using triangular multiplication.
 * 2. Computes $AtA = A^T \times A$ using triangular multiplication.
 * 3. Computes $Res = AB \times B^T + AtA$ using general matrix multiplication.
 * 
 * Time Complexity: $O(N^3)$ dominated by BLAS kernels.
 * Space Complexity: $O(N^2)$ to store intermediate result matrices.
 */
double* my_solver(int N, double *A, double *B) {
	double *AB = malloc((N * N) * sizeof(double));
	copy(N, &AB, B);
	
	/**
	 * Functional Utility: In-place triangular matrix-matrix multiplication.
	 * Logic: Computes $B = \alpha \times A \times B$ where $A$ is upper triangular.
	 */
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				AB, N);

	double *AtA = malloc((N * N) * sizeof(double));
	copy(N, &AtA, A);
	
	/**
	 * Functional Utility: Computes $A^T \times A$ for triangular $A$.
	 */
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				AtA, N);

	double *res = malloc((N * N) * sizeof(double));
	copy(N, &res, AtA);
	
	/**
	 * Functional Utility: Accumulates the final product into the result matrix.
	 * Logic: Computes $C = 1.0 \times (AB \times B^T) + 1.0 \times C$.
	 */
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N, N, N, 
				1.0, 
				AB, N,
				B, N,
				1.0,
				res, N
				);

	free(AB);
	free(AtA);
	return res;
}
