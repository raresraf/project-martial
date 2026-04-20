/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

 #include "cblas.h"
#include "utils.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
    double *arr2 = calloc(N * N, sizeof(double));
    double *arr3 = calloc(N * N, sizeof(double));
    double *arr = calloc(N * N, sizeof(double));
    double *arr1 = calloc(N * N, sizeof(double));


    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(int i = 0; i < N ; i ++){
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int j=0; j < N; j++)
        	arr[i * N + j] = A[i * N + j];
    }

    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, arr, N);


    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(int i = 0; i < N ; i ++){
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int j=0; j < N; j++)
        	arr1[i * N + j] = B[i * N + j];
    }

    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, arr1, N);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, arr1, N, B, N, 1, arr2, N);

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(int i = 0; i < N * N; i++){
        arr3[i] = arr2[i] + arr[i];
    }

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(int i = 0; i < N ; i ++){
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int j=0; j < N; j++)
        	arr3[i * N + j] = arr2[i * N + j] + arr[i * N + j];
    }

    free(arr2);
    free(arr1);
    free(arr);

    return arr3;
}
