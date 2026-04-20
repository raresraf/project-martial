/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int n, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C = (double *)calloc(n * n, sizeof(double));
	double *D = (double *)calloc(n * n, sizeof(double));

	register int bi=0;
    register int bj=0;
    register int bk=0;
    register int i=0;
    register int j=0;
    register int k=0;

	int blockSize = 25;
	
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(bi = 0; bi < n; bi+=blockSize) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(bj = 0; bj < n; bj+=blockSize) {
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(bk = 0; bk < n; bk+=blockSize) {
				/**
				 * Block Logic: Iterative processing loop.
				 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
				 */
				for(i = 0; i < blockSize; i++) {
					
					register double *orig_pbt = &B[(bi+i) * n + bk]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

					/**
					 * Block Logic: Iterative processing loop.
					 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
					 */
					for(j = 0; j < blockSize; j++) {

						register double *pbt = orig_pbt; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
						register double *pb = &B[(bj + j) * n + bk]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

						register double suma = 0.0;

						/**
						 * Block Logic: Iterative processing loop.
						 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
						 */
						for(k = 0; k < blockSize; k++) {

							suma += *pbt * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

							pbt++;
							pb++;
						}
						D[(bi+i) * n + bj + j] += suma;
					}
				}
			}
		}
	}

	
	
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(bi = 0; bi < n; bi+=blockSize) {
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for(bj = 0; bj < n; bj+=blockSize) {
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for(bk = bi; bk < n; bk+=blockSize) {

				register int start = 0;

                /**
                 * Block Logic: Iterative processing loop.
                 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                 */
                for(i = 0; i < blockSize; i++) {
					
					register double *orig_pa;

					/**
					 * Block Logic: Conditional state branch.
					 * Invariant: The conditional branch maintains control flow invariants.
					 */
					if(bk + blockSize - bi - i < 0) {
						start = i;
						orig_pa = &A[(bi + i) * n + bk + i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					} else {
					    orig_pa = &A[(bi + i) * n + bk]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					}

                    /**
                     * Block Logic: Iterative processing loop.
                     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                     */
                    for(j = 0; j < blockSize; j++) {

						register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
   						register double *pd = &D[(bk  + start) * n + bj + j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

						register double suma = 0.0;
					
                        /**
                         * Block Logic: Iterative processing loop.
                         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                         */
                        for( k = start; k < blockSize; k++) {

						   suma += *pa * *pd; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

     					   pa++;
                           pd += n;
						}
						C[(bi+i) * n + bj + j] += suma;
					}

				}
			}
		}
	}

	
	
	 /**
	  * Block Logic: Iterative processing loop.
	  * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	  */
	 for(bi = 0; bi < n; bi+=blockSize) {
		register double end = bi;
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for(bj = bi; bj < n; bj+=blockSize) {
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for(bk = 0; bk < n; bk+=blockSize) {
                /**
                 * Block Logic: Iterative processing loop.
                 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                 */
                for(i = 0; i < blockSize; i++) {
					end++;
					register double *orig_pat = &A[bk * n + bi + i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					/**
					 * Block Logic: Iterative processing loop.
					 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
					 */
					for(j = 0; j < blockSize; j++) {

						/**
						 * Block Logic: Conditional state branch.
						 * Invariant: The conditional branch maintains control flow invariants.
						 */
						if (bj + j < bi + i) {
							continue;
						}
						
						register double *pat = orig_pat; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
   						register double *pa = &A[bk * n  + bj + j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
						
						register double suma = 0.0;

                        /**
                         * Block Logic: Iterative processing loop.
                         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
                         */
                        for(k = 0; k < blockSize && bk + k <= end; k++) {

                            suma +=  *pat * *pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

							pat += n;
                            pa += n;
						}
						/**
						 * Block Logic: Conditional state branch.
						 * Invariant: The conditional branch maintains control flow invariants.
						 */
						if (bi + i != bj + j) {
							C[(bj+j) * n + bi + i] += suma;
						} 
						C[(bi+i) * n + bj + j] += suma;
					}

				}
			}
		}
	}

	free(D);
	
	return C;	
}
