/**
 * @raw/aeb31e2b-0ab9-4296-9704-30003e34fa86/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = (A * B) * B^T + A^T * A.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C, *aux;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers tracking memory addresses appropriately.
	 */
	C = (double *)malloc(N * N * sizeof(double));
	aux = (double *)malloc(N * N * sizeof(double));

	/**
	 * Functional Utility: Duplicates matrix B inside array parameter aux to generate the formulation components utilizing standard Level 3 operations.
	 */
	cblas_dcopy(N * N, B, 1, aux, 1); 

	/**
	 * Functional Utility: Calculates the fundamental step determining aux = A * B leveraging native dtrmm mappings targeting upper triangular arrays.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, aux, N); 

	/**
	 * Functional Utility: Executes computation performing overall accumulation defining combination mapping directly into C.
	 * Executes C = aux * B^T where aux = A * B currently.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, aux, N, B, N, 0, C, N); 

	/**
	 * Functional Utility: Isolates memory generating A to define aux subsequently executing an in-place conversion transforming aux = A^T * A.
	 */
	cblas_dcopy(N * N, A, 1, aux, 1); 
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, aux, N); 

	/**
	 * Functional Utility: Synchronizes state combinations computing C = C + aux utilizing daxpy.
	 */
	cblas_daxpy(N * N, 1, aux, 1, C, 1); 

	free(aux);

	return C;
}
