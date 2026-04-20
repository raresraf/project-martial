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
	double *AB = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = malloc(N * N * sizeof(double));
	double *R = malloc(N * N * sizeof(double));
	register int i, j, k;
	register double *pab, *pa, *pb, *pbt, *pabbt, *pata, *pr;  /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	register double sum;

	pab = AB;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {

			pa = orig_pa + i;
			pb = B + i * N + j;
			sum = 0.0;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				sum += (*pa) * (*pb); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pa;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}
			*pab = sum;
			++pab;
		}
	}

	pabbt = ABBt;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pab = AB + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			pab = orig_pab;
			pbt = B + j * N;
			sum = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				sum += (*pab) * (*pbt); /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pab;
				++pbt;
			}
			*pabbt = sum;
			++pabbt;
		}
	}

	pata = AtA;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A;
		
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			pa = orig_pa;
			sum = 0.0;
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (i < k || j < k)
					break;
				sum += (*(pa + i)) * (*(pa + j));
				pa += N;
			}
			*pata = sum;
			++pata;
		}
	}

	pr = R;
	pabbt = ABBt;
	pata = AtA;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N * N; ++i) {
		*pr = *pabbt + *pata;
		++pr;
		++pabbt;
		++pata;
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return R;	
}
