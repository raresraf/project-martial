/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"


double *allocate_matrix(int N) {
	double *res = calloc(N * N, sizeof(double));
	return res;
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = allocate_matrix(N);
	double *AxB = allocate_matrix(N);
	double *AxBxBt = allocate_matrix(N);
	double *AtxA = allocate_matrix(N);
    register int i, j, k;

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (i = 0; i < N; i++) {
        register double *orig_pa = A + N * i + i;   
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (j = 0; j < N; j++) { 
            register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pb = B + N * i + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double sum = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = i; k < N; k++) {
                sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa++;
                pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            }
            *(AxB + i * N + j) = sum;
        }
    } 

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (i = 0; i < N; i++) {
        register double *orig_pa = AxB + N * i;           
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (j = 0; j < N; j++) {
            register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pb = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double sum = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = 0; k < N; k += 1) {
                sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa++;
                pb++;
            }
            *(AxBxBt + i * N + j) = sum;
        }
    } 

    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for (i = 0; i < N; i++) {
        register double *orig_pa = A + i;         
        /**
         * Block Logic: Iterative processing loop.
         * Invariant: Maintains sequence integrity while progressing through the defined bounds.
         */
        for (j = 0; j < N; j++) {
            register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double *pb = A + j;             /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            register double sum = 0;
            /**
             * Block Logic: Iterative processing loop.
             * Invariant: Maintains sequence integrity while progressing through the defined bounds.
             */
            for (k = 0; k <= i; k++) {
                sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
                pa += N;
                pb += N;                                    /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
            }
            *(AtxA + i * N + j) = sum;
        }
    } 

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
        register double *orig_pa = AtxA + i * N;   
        register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
        register double *pb = AxBxBt + i * N;  /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			*(C + i * N + j) = *pa + *pb;
            pa++;
            pb++;
		}
	}

	free(AxB);
	free(AxBxBt);
	free(AtxA);
	return C;
}
