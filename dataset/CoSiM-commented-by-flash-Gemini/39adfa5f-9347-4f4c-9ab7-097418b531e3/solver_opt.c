
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized implementation of the matrix expression solver.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = A * (B * B^T) + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs multiple manual C tuning techniques:
 * 1. Loop Unrolling: Processes 2x2 blocks of the output matrix per inner loop 
 *    iteration to reduce branch overhead and exploit ILP.
 * 2. Register Caching: Keeps frequently accessed accumulators and base pointers 
 *    in CPU registers.
 * 3. Pointer Arithmetic: Replaces array indexing with direct increments to 
 *    minimize address calculation latency.
 * 4. Reuse-Aware Access: Structures loops to maximize temporal and spatial 
 *    locality for the upper triangular matrix A.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized matrix solver using loop unrolling and pointer arithmetic.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

    /**
     * Block Logic: Compute BBt = B * B^T.
     * Optimization: 2x2 Blocking and Loop Unrolling.
     * Logic: Simultaneously computes four elements of the symmetric product 
     * using four register-backed accumulators (sum1..sum4).
     */
	double *BBt = (double *)calloc(N * N, sizeof(double));
	for (int i = 0; i < N; i += 2) {
	    register int row = i * N;
        register double *original_b = &(B[row]);
        register double *bbt = &(BBt[row]);

		for (int j = 0; j < N; j += 2) {
            register double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
            register double *b1 = original_b;
            register double *b2 = &(B[j * N]);

            /**
             * Block Logic: Dot product core.
             * Invariant: Computes partial sums for B[i]*B[j], B[i+1]*B[j], 
             * B[i]*B[j+1], and B[i+1]*B[j+1] in a single pass over k.
             */
			for (int k = 0; k < N; k++) {
				sum1 += (*b1) * (*b2);
                sum2 += (*(b1 + N)) * (*b2);
                sum3 += (*b1) * (*(b2 + N));
                sum4 += (*(b1 + N)) * (*(b2 + N));
                b1++;
                b2++;
			}

            // Serialization: Writes the 2x2 result block back to the workspace.
            *(bbt) = sum1;
            *(bbt + N) = sum2;
            *(bbt + 1) = sum3;
            *(bbt + N + 1) = sum4;
            bbt += 2;
		}
	}

    /**
     * Block Logic: Compute A_BBt = A * BBt.
     * Optimization: Triangular blocking.
     * Logic: Exploits the row-major storage and triangular property of A. 
     * Uses unrolling to propagate A[i][k] and A[i+1][k] across the BBt matrix.
     */
    double *A_BBt = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i += 2) {
        register int row = i * N;
        register double *a = &(A[row + i]);

		for (int k = i; k < N; k++) {
            register double *bbt = &(BBt[k * N]);
            register double *a_bbt = &(A_BBt[row]);

			for (int j = 0; j < N; j += 2) {
                *(a_bbt) += (*a) * (*bbt);
				*(a_bbt + N) += (*(a + N)) * (*bbt);
                *(a_bbt + 1) += (*a) * (*(bbt + 1));
                *(a_bbt + N + 1) += (*(a + N)) * (*(bbt + 1));
				bbt += 2;
                a_bbt += 2;
			}
			a++;
		}
	}

    /**
     * Block Logic: Compute AtA = A^T * A.
     * Optimization: Pointer-based transposition.
     * Logic: Accesses columns of A (as rows of A^T) via vertical pointer 
     * increments to generate the symmetric product.
     */
    double *AtA = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
		register int row =  i * N;

		for (int k = 0; k <= i; k++) {
            register double *a2 = &(A[k * N]);
            register double *a1 = a2 + i;
            register double *ata = &(AtA[row]);

			for (int j = 0; j < N; j++) {
				*(ata) += (*a1) * (*a2);
                a2++;
                ata++;
			}
		}
	}

    /**
     * Block Logic: Final unrolled aggregation pass.
     * Invariant: Sums the two component matrices in 2x2 blocks to match the 
     * optimization pattern of previous steps.
     */
    for (int i = 0; i < N; i += 2) {
        register int row = i * N;
        register double *ata = &(AtA[row]);
        register double *a_bbt = &(A_BBt[row]);

		for (int j = 0; j < N; j += 2) {
			*(a_bbt) += (*ata);
            *(a_bbt + N) += (*(ata + N));
            *(a_bbt + 1) += (*(ata + 1));
            *(a_bbt + N + 1) += (*(ata + N + 1));
            ata += 2;
            a_bbt += 2;
		}
	}

    free(AtA);
    free(BBt);
	return A_BBt;
}
