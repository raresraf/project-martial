/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"


#define DIE(assertion, call_description)				\
	do {								\
		/**
		 * Block Logic: Conditional state branch.
		 * Invariant: The conditional branch maintains control flow invariants.
		 */
		if (assertion) {					\
			fprintf(stderr, "(%s, %d): ",			\
					__FILE__, __LINE__);		\
			perror(call_description);			\
			exit(EXIT_FAILURE);				\
		}							\
	} while (0)




/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	double *M;
	register double *ATA;
	register double *AB;
	register double *ABBT;
	int i, j, k;

	M = (double *)malloc(N * N * sizeof(double));
	DIE(M == NULL, "malloc M");

	ATA = (double *)malloc(N * N * sizeof(double));
	DIE(ATA == NULL, "malloc ATA");

	AB = (double *)malloc(N * N * sizeof(double));
	DIE(AB == NULL, "malloc AB");

	ABBT = (double *)malloc(N * N * sizeof(double));
	DIE(ABBT == NULL, "malloc ABBT");

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pa = A + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = B + j + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k++) {
				suma += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}
			AB[i * N + j] = suma;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pa = AB + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = B + j * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k++) {
				suma += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa++;
				pb++;
			}
			ABBT[i * N + j] = suma;
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
			register double *pb = A + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
           		 /**
           		  * Block Logic: Iterative processing loop.
           		  * Invariant: Maintains sequence integrity while progressing through the defined bounds.
           		  */
           		 for (k = 0; k < i + 1; k++) {
                		suma += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pa += N;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
        		}
			ATA[i * N + j] += suma;
	    	}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++) {
			M[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
		}
	}

	free(ABBT);
	free(ATA);
	free(AB);

	return M;	
}
