/**
 * @raw/99113ad8-03b9-45a3-a87a-1ec710110a68/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Tiling optimization using sub-matrices explicitly designed to minimize L1/L2 cache misses.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches.
	 */
	double *C1 = calloc(N*N, sizeof(double));
	double *C = calloc(N*N, sizeof(double));

	int bi=0;
    int bj=0;
    int bk=0;
    int i=0;
    int j=0;
    int k=0;

	/**
	 * Inline: Tiling size specifically bounded to match optimal target CPU cache parameters.
	 */
    int blockSize=100; 
 
	/**
	 * Block Logic: Tiled iteration logic evaluating C1 = A * B.
	 * Optimization: Processes block by block reducing cache evictions. Uses local registers for inner products.
	 */
    for(bi=0; bi<N; bi+=blockSize){
        for(bj=0; bj<N; bj+=blockSize){
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++){
						register double tmp = 0;
                        for(k=0; k<blockSize; k++){
							if(bi+i > bk+k)
								continue;
                        	tmp += A[(bi+i)*N + bk+k] * B[(bk+k)*N + bj+j];
						}
						C1[(bi+i)*N + bj+j] += tmp;
					}
				}
			}
		}
	}

	/**
	 * Block Logic: Evaluates tiled execution computing C = C1 * B^T.
	 * Optimization: Sub-matrices traverse row arrays preserving temporal cache locality for transposed matrices.
	 */
	for(bi=0; bi<N; bi+=blockSize){
        for(bj=0; bj<N; bj+=blockSize){
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++){
						register double tmp = 0;
                        for(k=0; k<blockSize; k++){
                            tmp += C1[(bi+i)*N + bk+k] * B[(bj+j)*N + bk+k];
						}
						C[(bi+i)*N + bj+j] += tmp;
					}
				}
			}
		}
	}

	/**
	 * Block Logic: Updates C = C + A^T * A.
	 * Optimization: Nested iterations block-level bounding ensures constrained traversal space conforming to upper-triangular definitions.
	 */
	for(bi=0; bi<N; bi+=blockSize){
        for(bj=0; bj<N; bj+=blockSize){
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++){
						register double tmp = 0;
                        for(k=0; k<blockSize; k++){
							if(bi+i < bk+k)
								continue;
                            tmp += A[(bk+k)*N + bi+i] * A[(bk+k)*N + bj+j];
						}
						C[(bi+i)*N + bj+j] += tmp;
					}
				}
			}
		}
	}

	free(C1);

	return C;
}
