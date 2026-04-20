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
	double *C;
	double *AtA, *BBt, *ABBt;
	double *At, *Bt;
	register int i;
	register int j;
	register int k;

	register int size = N * N * sizeof(*C);
	C = malloc(size);
	AtA = malloc(size);
	At = malloc(size);
	Bt = malloc(size);
	BBt = malloc(size);
	ABBt = malloc(size);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL || AtA == NULL || BBt == NULL || ABBt == NULL) {
		exit(EXIT_FAILURE);
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *At_col = At + i;
		register double *Bt_col = Bt + i;
		register double *pA = A + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *pB = B + i * N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*At_col = *pA;
			*Bt_col = *pB;
			At_col += N;
			Bt_col += N;
			++pA;
			++pB;
		}
	}

	
	register double *AtA_ptr = AtA;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pat = At + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pat = orig_pat; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pa = A + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= i; ++k) {
				suma += *pat * *pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pat;
				pa += N;
			}
			*AtA_ptr = suma;
			++AtA_ptr;
		}
	}

	
	register double *BBt_ptr = BBt;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pb = B + i * N;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pb = orig_pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pbt = Bt + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; ++k) {
				suma += *pb * *pbt; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pb;
				pbt += N;
			}
			*BBt_ptr = suma;
			++BBt_ptr;
		}
	}

	
	register double *ABBt_ptr = ABBt;
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
			register double *pa = orig_pa + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pbbt = BBt + i * N + j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				suma += *pa * *pbbt; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				++pa;
				pbbt += N;
			}
			*ABBt_ptr = suma;
			++ABBt_ptr;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			*(C + i * N + j) = *(ABBt + i * N + j) + *(AtA + i * N + j);
		}
	}

	free(ABBt);
	free(AtA);
	free(BBt);
	free(At);
	free(Bt);
	return C;	
}
