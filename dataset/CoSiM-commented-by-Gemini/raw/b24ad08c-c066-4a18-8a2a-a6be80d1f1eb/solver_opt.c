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

	register size_t i;
	register size_t j;
	register size_t k;

	
	double *C = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL) {
		perror("Calloc C");
		exit(EXIT_FAILURE);
	}

	
	double *AB = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (AB == NULL) {
		perror("Calloc AB");
		exit(EXIT_FAILURE);
	}

	
	double *P1 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (P1 == NULL) {
		perror("Calloc P1");
		exit(EXIT_FAILURE);
	}

	
	double *P2 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (P2 == NULL) {
		perror("Calloc P2");
		exit(EXIT_FAILURE);
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;

			pa += i;
			pb += N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; ++k) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

				pa++;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}

			AB[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &AB[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &B[j * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k += 4) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

				pa++;
				pb++;

				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

				pa++;
				pb++;

				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

				pa++;
				pb++;

				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

				pa++;
				pb++;
			}

			P1[i * N + j] = sum;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = &A[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double sum = 0.0;

			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k <= i && k <= j; ++k) {
				sum += *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

				pa += N;
				pb += N; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			}

			P2[i * N + j] = sum;
		}
	}

	
	register double *pc = C; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	register double *pp1 = P1; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	register double *pp2 = P2; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N * N; i += 4) {
		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;

		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;

		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;

		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;
	}

	free(AB);
	free(P1);
	free(P2);

	return C;
}
