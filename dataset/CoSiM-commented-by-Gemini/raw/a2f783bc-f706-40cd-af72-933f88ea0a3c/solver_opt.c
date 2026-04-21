/**
 * @file solver_opt.c
 * @brief Performance-optimized matrix operation solver utilizing cache-aware tiling and pointer arithmetic.
 * 
 * Architectural Intent: Implements a high-performance linear algebra routine designed to maximize 
 * CPU throughput and cache utilization. It uses blocking (tiling) to ensure that sub-matrices 
 * fit within the L1/L2 cache boundaries, minimizing memory stalls.
 * 
 * Performance Strategies:
 * - Cache Tiling: Block size of 40x40.
 * - Register Caching: Hints the compiler to maintain active pointers in CPU registers.
 * - Optimized Memory Access: Linear pointer increments replace multi-dimensional indexing.
 */

#include "utils.h"

void wait_for_input(const char *msg)
{
	printf("%s\n", msg);
	getc(stdin);
}

/**
 * @brief Computes a complex matrix expression with optimized blocked multiplication.
 * Algorithm: Blocked (Tiled) Matrix Multiplication.
 * Time Complexity: $O(N^3)$ with reduced constant factors due to cache hits.
 * Space Complexity: $O(N^2)$ for intermediate buffers.
 */
double* my_solver(int N, double *A, double* B)
{
	double *res = calloc(N * N, sizeof(*res));
	double *tmp1 = calloc(N * N, sizeof(*res));
	double *tmp2 = calloc(N * N, sizeof(*res));
	int blockSize = 40;

	if (res == NULL || tmp1 == NULL || tmp2 == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	/**
	 * Block Logic: Tiled matrix multiplication pass.
	 * Pre-condition: Input matrices A and B are valid and contiguous.
	 * Invariant: Outer loops traverse blocks, inner loops perform dense multiplication within tiles.
	 */
	for (register int bi = 0; bi < N; bi += blockSize) {
		for (register int bj = bi; bj < N; bj += blockSize) {
			for (register int bk = 0; bk < N; bk += blockSize) {

				/**
				 * Block Logic: Conditional optimization for sub-block boundaries.
				 */
				if (bk < bi + blockSize) {
					for (register int i = 0; i < blockSize; i++) {
						register double *orig_c = &tmp1[(bi + i) * N + bj]; 
						register double *orig_a = &B[(bi + i) * N + bk]; 
						register double *orig_b = &B[bj * N + bk]; 

						register double *orig_c2 = &tmp2[(bi + i) * N + bj]; 
						register double *orig_a2 = &A[bk * N + bi + i]; 
						register double *orig_b2 = &A[bk * N + bj]; 

						/**
						 * Block Logic: Inner dot-product accumulation.
						 * Invariant: Uses pointer arithmetic to step through rows and columns without re-calculating offsets.
						 */
						for (register int k = 0; k < blockSize; k++) {
							register double *pc1 = orig_c;
							register double *pa1 = orig_a + k;
							register double *pb1 = orig_b + k;

							if (bi + i >= bk + k) {
								register double *pa2 = orig_a2 + k * N;
								register double *pc2 = orig_c2;
								register double *pb2 = orig_b2 + k * N;
								for (register int j = 0; j < blockSize; j++) {
									// Functional Utility: Fused multiply-add for first product set.
									*pc1 += *pa1 * *pb1;
									pc1++;
									pb1 += N;

									// Functional Utility: Fused multiply-add for second product set.
									*pc2 += *pa2 * *pb2;
									pc2++;
									pb2++;
								}
							} else {
								for (register int j = 0; j < blockSize; j++) {
									*pc1 += *pa1 * *pb1;
									pc1++;
									pb1 += N;
								}
							}
						}
					}
				} else {
					for (register int i = 0; i < blockSize; i++) {
						register double *orig_c = &tmp1[(bi + i) * N + bj]; 
						register double *orig_a = &B[(bi + i) * N + bk]; 
						register double *orig_b = &B[bj * N + bk]; 

						for (register int k = 0; k < blockSize; k++) {
							register double *pc1 = orig_c;
							register double *pa1 = orig_a + k;
							register double *pb1 = orig_b + k;

							for (register int j = 0; j < blockSize; j++) {
									*pc1 += *pa1 * *pb1;
									pc1++;
									pb1 += N;
							}
						}
					}
				}
			}
		}
	}

	/**
	 * Block Logic: Symmetry correction and matrix transposition.
	 */
	for (int i = 0; i < N ; i++) {
		for (int j = 0; j < N; j++) {
			tmp2[j * N + i] = tmp2[i * N + j];
			tmp1[j * N + i] = tmp1[i * N + j];
		}
	}

	// ... (Rest of matrix addition and cleanup) ...
	free(tmp1);
	free(tmp2);

	return res;
}
