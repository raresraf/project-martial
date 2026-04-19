/**
 * @raw/88e62bc5-c880-450e-9891-e1500da66061/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = (A * B) * B^T + A^T * A using SGEMM derivatives.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */
#include "utils.h"
#include <string.h>
#include <cblas.h>

/**
 * Inline: Pruning bounds computation macro (though unused primarily due to BLAS primitives).
 */
#define min(x, y) (((x) < (y)) ? (x) : (y))

double* my_solver(int N, double *A, double *B) {
	
	/**
	 * Functional Utility: Initializes tracking memory buffers and configures 
	 * the operands for subsequent in-place BLAS execution.
	 */
	double *C = (double *)malloc(N * N * sizeof(double));
        double *term = (double *)malloc(N * N * sizeof(double));

	memcpy(term, B, N * N * sizeof(double));
	
	/**
	 * Functional Utility: Computes term = A * B.
	 * Targets the upper triangular form of matrix A dynamically.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, term, N);	

	/**
	 * Functional Utility: Computes A = A^T * A in-place.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);

	memcpy(C, A, N * N * sizeof(double));
	
	/**
	 * Functional Utility: Computes final aggregation C = C + term * B^T = A^T * A + (A * B) * B^T.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, term, N, B, N, 1, C, N);

        free(term);

        return C;
}
