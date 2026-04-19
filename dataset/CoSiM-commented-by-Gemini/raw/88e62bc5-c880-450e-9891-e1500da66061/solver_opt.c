/**
 * @raw/88e62bc5-c880-450e-9891-e1500da66061/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Matrix multiplication employing pointer arithmetic and CPU registers for sum aggregation.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */
#include "utils.h"

/**
 * Inline: Evaluates the minimum boundary to prune matrix multiplication loops dynamically.
 */
#define min(x, y) (((x) < (y)) ? (x) : (y))

double* my_solver(int N, double *A, double* B) {

        double *C = (double *)malloc(N * N * sizeof(double));
        double *term = (double *)malloc(N * N * sizeof(double));
	register double sum = 0.0;
	register double *orig_a, *a, *b;
	register int i, j, k, aux;

	/**
	 * Block Logic: Computes term = A * B.
	 * Optimization: Employs explicit pointer tracking to evade multiplication overheads during indexing.
	 */
        for (i = 0;i < N;i++) {
		aux = i * N;
                orig_a = A + (aux + i);
		for (j = 0;j < N;j++) {
			a = orig_a;
			b = B + (aux + j);
                        sum = 0.0;
			for (k = i;k < N;k++) {
                                sum += *a * *b;
				a++;
				b += N;
                        }
			term[aux + j] = sum;
                }
        }

	/**
	 * Block Logic: Computes C = term * B^T.
	 * Optimization: Accesses B in sequential memory chunks simulating transposed layout.
	 */
        for (i = 0;i < N;i++) {
		aux = i * N;
		orig_a = term + aux;
                for (j = 0;j < N;j++) {
			a = orig_a;
			b = B + (j * N);
                        sum = 0.0;
                        for (k = 0;k < N;k++) {
                                sum += *a * *b;
        			a++;
				b++; 
	               }  
			C[aux + j] = sum;
                }
        }

	/**
	 * Block Logic: Computes term = A^T * A.
	 * Optimization: Strides the internal pointer arrays vertically while skipping trailing zeroes.
	 */
        for (i = 0;i < N;i++) {
        	aux = i * N;
		orig_a = A + i;     
		for (j = 0;j < N;j++) {
                        a = orig_a;
			b = A + j;
			sum = 0.0;
                        for (k = 0;k <= min(i, j);k++) {
                                sum += *a * *b;
                        	a += N;
				b += N;
			}
			term[aux + j] = sum;
		}
	}

	/**
	 * Block Logic: Final summation step synthesizing C += term.
	 * Invariant: Aggregates computed matrices into a cohesive single array.
	 */
	for (i = 0;i < N;i++) {
		aux = i * N;
		orig_a = term + aux;
                for (j = 0;j < N;j++) {
                        C[aux + j] += *orig_a;
			orig_a++;
                }
        }

        free(term);

	return C;	
}
