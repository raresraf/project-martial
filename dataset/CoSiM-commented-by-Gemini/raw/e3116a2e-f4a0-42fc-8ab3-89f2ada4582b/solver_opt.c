/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C = (double *)calloc(N*N,sizeof(double));
	
	double *AUX = (double *)calloc(N*N, sizeof(double));

	register int bi=0;
    register int bj=0;
    register int bk=0;
    register int i=0;
    register int j=0;
    register int k=0;
	register int blockSize = 40;
    
    /**
     * Block Logic: Blocked matrix multiplication for $AUX = A \times B$.
     * Invariant: Computes partial sums localized to blockSize x blockSize chunks to maximize L1 cache hit rate.
     */
	for(bi=0; bi<N; bi+=blockSize)
	{
        for(bj=0; bj<N; bj+=blockSize)
		{
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; ++i)
				{
                    for(j=0; j<blockSize; ++j)
					{
						register double res = 0;
                        for(k=0; k<blockSize; ++k){
                            // Inline: Skips explicit zero elements from upper triangular structure of matrix A.
                            if(bi + i<=bk + k)
							{
								res += A[((bi+i) * N) + (bk+k)] * B[((bk+k) * N) + (bj+j)];
							}
						}
						AUX[((bi+i) * N) + (bj+j)] += res;
					}
				}
			}
		}
	}

    /**
     * Block Logic: Blocked matrix evaluation for $C = AUX \times B^T + A^T \times A$.
     * Invariant: Accumulates outputs for target block bounds, reusing data cached across nested iterations.
     */
	for(bi=0; bi<N; bi+=blockSize)
        for(bj=0; bj<N; bj+=blockSize)
            for(bk=0; bk<N; bk+=blockSize)
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++)
					{
						register double res = 0;
                        for(k=0; k<blockSize; k++){
							res += AUX[((bi+i) * N) + (bk+k)] * B[((bj+j) * N) + (bk+k)];
                            if(bk + k<=bj + j){
								res += A[((bk+k) * N) + (bi+i)] * A[((bk+k) * N) + (bj+j)];
							}
						}
						C[((bi+i) * N) + (bj+j)] += res;
					}
				}
	free(AUX);
	return C;
}
