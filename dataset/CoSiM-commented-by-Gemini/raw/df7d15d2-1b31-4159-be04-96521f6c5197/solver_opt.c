/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"
#include <string.h>
#include <immintrin.h>
#define min(a,b) (((a)<(b))?(a):(b))

void allocate_memory(int N, double **C, double **A_t, double **B_t,
					double **left_side, double **right_side, double **tmp)
{
	*C = aligned_alloc(32, N * N * sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*C)) {
		exit(-1);
	}

	*A_t = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*A_t)) {
		exit(-1);
	}

	*B_t = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*B_t)) {
		exit(-1);
	}

	*left_side = (double *)aligned_alloc(32, N * N * sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*left_side)) {
		exit(-1);
	}

	*right_side = (double *)aligned_alloc(32, N * N * sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*right_side)) {
		exit(-1);
	}

	*tmp = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (!(*tmp)) {
		exit(-1);
	}
}

void avx_add(double **result, const double *a, const double *b, size_t N)
{
	register size_t i = 0;

#ifdef __AVX512__
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (; i < (N & ~0x7); i += 8) {
		const __m512d kA8 = _mm512_load_pd(&a[i]);
		const __m512d kB8 = _mm512_load_pd(&b[i]);

		const __m512d kRes = _mm512_add_pd(kA8, kB8);
		_mm512_stream_pd(&(*result)[i], kRes);
	}
#endif
#ifdef __AVX__
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (; i < (N & ~0x3); i += 4) {
		const __m256d kA4 = _mm256_load_pd(&a[i]);
		const __m256d kB4 = _mm256_load_pd(&b[i]);

		const __m256d kRes = _mm256_add_pd(kA4, kB4);
		_mm256_stream_pd(&(*result)[i], kRes);
	}
#endif
#ifdef __SSE__
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (; i < (N & ~0x1); i += 2) {
		const __m128d kA2 = _mm_load_pd(&a[i]);
		const __m128d kB2 = _mm_load_pd(&b[i]);

		const __m128d kRes = _mm_add_pd(kA2, kB2);
		_mm_stream_pd(&(*result)[i], kRes);
	}
#endif

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (; i < N; i++) {
		*result[i] = a[i] + b[i];
	}
}


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	register size_t i, j, k;
	double *C, *A_t, *B_t, *left_side, *right_side, *tmp;

	allocate_memory(N, &C, &A_t, &B_t, &left_side, &right_side, &tmp);

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *pa_t = A_t + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *pb_t = B_t + i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *pa = A + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *pb = B + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j, pa_t += N, pb_t += N, pa++, pb++) {
			*pa_t = *pa;
			*pb_t = *pb;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *pleft_side = left_side + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *og_pa = A + N * i;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa = og_pa + i; 		 /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = B_t + N * j + i; 	 /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			register int cnt = i;
			
			
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = i; k < N; k += 4, pa += 4, pb += 4) {
				suma = suma + *pa * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				cnt++;
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < N) {
					suma = suma + *(pa + 1) * *(pb + 1);
					cnt++;
				} else {
					break;
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < N) {
					suma = suma + *(pa + 2) * *(pb + 2);
					cnt++;
				} else {
					break;
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < N) {
					suma = suma + *(pa + 3) * *(pb + 3);
					cnt++;
				} else {
					break;
				}
			}
			*pleft_side = suma;
			pleft_side++;
		}
	}

	
	memcpy(tmp, left_side, N * N * sizeof(double));

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *pleft_side = left_side + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *og_tmp = tmp + N * i;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *ptmp = og_tmp; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pb = B + N * j;  /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			register int cnt = 0;
			
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < N; k += 4, ptmp += 4, pb += 4) {
				suma = suma + *ptmp * *pb; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				cnt++;
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < N) {
					suma = suma + *(ptmp + 1) * *(pb + 1);
					cnt++;
				} else {
					break;
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < N) {
					suma = suma + *(ptmp + 2) * *(pb + 2);
					cnt++;
				} else {
					break;
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < N) {
					suma = suma + *(ptmp + 3) * *(pb + 3);
					cnt++;
				} else {
					break;
				}
			}
			*pleft_side = suma;
			pleft_side++;
		}
	}

	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		register double *pright_side = right_side + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		register double *og_pa_t = A_t + N * i;
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			register double *pa_t = og_pa_t; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double *pa = A_t + N * j; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			register int cnt = 0;
			register size_t mn = min(j + 1, i + 1);
			
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for (k = 0; k < mn; k += 4, pa_t += 4, pa += 4) {
				suma = suma + *pa_t * *pa; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
				cnt++;
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < mn) {
					suma = suma + *(pa_t + 1) * *(pa + 1);
					cnt++;
				} else {
					break;
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < mn) {
					suma = suma + *(pa_t + 2) * *(pa + 2);
					cnt++;
				} else {
					break;
				}
				/**
				 * Block Logic: Conditional state branch.
				 * Invariant: The conditional branch maintains control flow invariants.
				 */
				if (cnt < mn) {
					suma = suma + *(pa_t + 3) * *(pa + 3);
					cnt++;
				} else {
					break;
				}
			}
			*pright_side = suma;
			pright_side++;
		}
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		double *pc = C + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		double *pleft_side = left_side + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		double *pright_side = right_side + N * i; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		avx_add(&pc, pleft_side, pright_side, N);
	}

	free(left_side);
	free(right_side);
	free(tmp);
	free(A_t);
	free(B_t);

	return C;
}
