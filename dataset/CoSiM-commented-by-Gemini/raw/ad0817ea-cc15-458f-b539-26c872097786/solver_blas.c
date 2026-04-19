/**
 * @raw/ad0817ea-cc15-458f-b539-26c872097786/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A^T * A + (A * B) * B^T.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {

	double *res1, *res2;
	
	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers tracking memory addresses appropriately.
	 */
	res1 = malloc(N * N * sizeof(double));
	res2 = malloc(N * N * sizeof(double));

	/**
	 * Functional Utility: Calculates A^T * A natively.
	 */
	dgemm("T","N",N,N,N,1,A,N,A,N,0,res1,N);

	/**
	 * Functional Utility: Calculates A * B.
	 */
	dgemm("N","N",N,N,N,1,A,N,B,N,0,res2,N);

	/**
	 * Functional Utility: Executes computation performing overall accumulation defining combination mapping directly into res1.
	 * Executes res1 = res1 + res2 * B^T where res2 = A * B currently.
	 */
	dgemm("N","T",N,N,N,1,res2,N,B,N,1,res1,N);

	free(res2);

	return res1;

	printf("BLAS SOLVER\n");
	return NULL;
}
