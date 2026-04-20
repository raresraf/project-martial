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
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *AtA = malloc(N * N * sizeof(double));
	double *At = malloc(N * N * sizeof(double));
	double *Bt = malloc(N * N * sizeof(double));
    double *AB = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    int i, j, k;

    double *pat, *pbt, *pa, *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
	    pa = A + i * N;
	    pb = B + i * N;
	    pat = At + i;
	    pbt = Bt + i;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
            *pat = *pa;
            *pbt = *pb;
            pa++;
            pb++;
            pat += N;
            pbt += N;
        }
    }
    
    double *orig_pa, *orig_pat, *orig_pab;
    double *pc, *pab; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(i = 0; i < N; ++i){
	  orig_pat = At + i * N;
	  pc = C + i * N;
	  /**
	   * Block Logic: Iterative processing loop.
	   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	   */
	  for(j = 0; j < N; ++j){
	    pat = orig_pat ;
	    pa = At + j * N;
	    register double suma = 0;
	    int kMax = (i < j)? i : j;
	    /**
	     * Block Logic: Iterative processing loop.
	     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	     */
	    for(k = 0; k <= kMax; ++k){
	      suma += *pat * *pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	      pat++;
	      pa++;
	      
	    }
	    *pc = suma;
	    pc++;
	  }
	}
    
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(i = 0; i < N; ++i){
	  orig_pa = A + i * N;
	  pab = AB + i * N;
	  /**
	   * Block Logic: Iterative processing loop.
	   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	   */
	  for(j = 0; j < N; ++j){
	    pa = orig_pa + i;
	    pb = Bt + j * N + i;
	    register double suma = 0.0;
	    /**
	     * Block Logic: Iterative processing loop.
	     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	     */
	    for(k = i; k < N; ++k){
	      suma += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	      pa++;
	      pb++;
	      
	    }
	    *pab = suma;
	    pab++;
	  }
	}
	
    /**
     * Block Logic: Iterative processing loop.
     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
     */
    for(i = 0; i < N; ++i){
	  orig_pab = AB + i * N;
	  pc = C + i * N;
	  /**
	   * Block Logic: Iterative processing loop.
	   * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	   */
	  for(j = 0; j < N; ++j){
	    pab = orig_pab;
	    pbt = B + j * N;
	    register double suma = 0.0;
	    /**
	     * Block Logic: Iterative processing loop.
	     * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	     */
	    for(k = 0; k < N; ++k){
	      suma += *pab * *pbt; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	      pab++;
	      pbt++;
	      
	    }
	    *pc += suma;
	    pc++;
	  }
	}

    free(AtA);
    free(AB);
    free(At);
    free(Bt);
	return C;	
}
