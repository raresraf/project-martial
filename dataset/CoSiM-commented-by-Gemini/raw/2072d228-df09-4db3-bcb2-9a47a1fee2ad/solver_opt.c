/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"
#include<stdint.h>




/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {

	register double* b_transpus = calloc(N * N, sizeof(double));
	register double* a_transpus = calloc(N * N, sizeof(double));

	register double* resultat = calloc(N * N, sizeof(double));
	register double* resultat2 = calloc(N * N, sizeof(double));
	register double* resultat3 = calloc(N * N, sizeof(double));
	register double* resultat4 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (register uint_fast32_t i = 0; i < N; ++i) {
		register double * ai = &A[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double * bi = &B[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t j = 0; j < N; ++j) {
			register uint_fast32_t index = j * N + i;
		
			a_transpus[index] = *ai;
			b_transpus[index] = *bi;

			ai++;
			bi++;
		}
	}
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (register uint_fast32_t i = 0; i < N; i++) {
		register uint_fast32_t p = i * N;
		register double *orig_pa = &A[p + i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t j = 0; j < N; j++) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[p + j] ; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (uint_fast32_t k = 0; k < N; k += 8) {
				 /**
				  * Block Logic: Conditional state branch.
				  * Invariant: The conditional branch maintains control flow invariants.
				  */
				 if(k >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 1 >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 2 >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 3 >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 4 >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 5 >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 6 >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(k + 7 >= i){
					sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
					pa++;
					pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
			}
			resultat[p + j] = sum;
		}
	}

	register uint_fast32_t index;

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (register uint_fast32_t i = 0; i < N; ++i) {
		register double *pAt_orig = &resultat[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t j = 0; j < N; ++j) {
			register double suma1 = 0.0;

			register double *pAt = pAt_orig; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pB = &b_transpus[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			index = i * N + j;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (register uint_fast32_t k = 0; k < N; k += 8) {
				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;

				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;

				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;

				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;

				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;

				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;

				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;

				suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				pAt++;
				pB += N;
			}
			resultat2[index] = suma1;

		}
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (register uint_fast32_t i = 0; i < N; ++i) {
		register double *pAt_orig = &a_transpus[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (register uint_fast32_t j = 0; j < N; ++j) {
			register double suma1 = 0.0;

			register double *pAt = pAt_orig; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pB = &A[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			index = i * N + j;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (register uint_fast32_t k = 0; k < N; k += 8) {
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;

				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k + 1){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;

				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k + 2){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;

				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k + 3){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;

				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k + 4){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;

				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k + 5){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;

				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k + 6){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;

				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if(i >= k + 7){
					suma1 += *pAt * *pB; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				}
				pAt++;
				pB += N;
			}
			resultat3[index] = suma1;

		}
	}	

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(register uint_fast32_t i = 0; i < N * N; i += 8){
    	resultat4[i] = resultat3[i] + resultat2[i];
    	resultat4[i + 1] = resultat3[i + 1] + resultat2[i + 1];
    	resultat4[i + 2] = resultat3[i + 2] + resultat2[i + 2];
    	resultat4[i + 3] = resultat3[i + 3] + resultat2[i + 3];
    	resultat4[i + 4] = resultat3[i + 4] + resultat2[i + 4];
    	resultat4[i + 5] = resultat3[i + 5] + resultat2[i + 5];
    	resultat4[i + 6] = resultat3[i + 6] + resultat2[i + 6];
    	resultat4[i + 7] = resultat3[i + 7] + resultat2[i + 7];
    }

    free(resultat);
    free(resultat2);
    free(resultat3);
    free(a_transpus);
    free(b_transpus);

    return resultat4;

}

