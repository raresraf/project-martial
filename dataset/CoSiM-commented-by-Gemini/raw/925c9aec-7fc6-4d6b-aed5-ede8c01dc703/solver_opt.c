/**
 * @raw/925c9aec-7fc6-4d6b-aed5-ede8c01dc703/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A * B * B^T + A^T * A.
 * Algorithm: Cache-friendly loop reordering, explicit pointer incrementations, and symmetrical mirroring optimization.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to auxiliary storage required for transposed layout buffers.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	double *BBT, *ABBT, *C, *AT, *ATA;

	/**
	 * Functional Utility: Configures primary buffers guaranteeing clean aggregation points initialized to zero.
	 */
	BBT = calloc(N * N , sizeof(double));
	ABBT = calloc(N * N , sizeof(double));
	C = calloc(N * N, sizeof(double));
	AT = calloc(N * N, sizeof(double));
	ATA = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Computes BBT = B * B^T.
	 * Optimization: Employs explicit pointer tracking to evade multiplication overheads during indexing.
	 */
    for (register int i = 0; i < N; ++i) {
		register double *ptr_bt = &(B[i * N]);
		register double *ptr_b = &(B[i * N]);
      	for (register int j = i; j < N; ++j) {
			register double sum = 0;
			register double *ptr_b_cpy = ptr_b;
        	for (register int k = 0; k < N; ++k) {
            	sum += (*ptr_b_cpy) * (*ptr_bt);
				ptr_b_cpy++;
				ptr_bt++;
         	}
			BBT[j * N + i] = BBT[i * N + j] = sum;
      	}
   }

	/**
	 * Block Logic: Computes ABBT = A * BBT using the ikj loop sequence.
	 * Optimization: Strides pointer arrays to maximize cache line usage during inner iterations.
	 */
    for (register int i = 0; i < N; ++i) {
		register double *ptr_a = &(A[i * N + i]);
		register double *ptr_bbt = &(BBT[i * N]);
		register double *ptr_abbt = &(ABBT[i * N]);
        for (register int k = i; k < N; ++k) {
			register double *ptr_bbt_cpy = ptr_bbt + (k - i) * N;
			register double *ptr_abbt_copy = ptr_abbt;
      		for (register int j = 0; j < N; ++j) {
            	(*ptr_abbt_copy) += (*ptr_a) * (*ptr_bbt_cpy);
				ptr_bbt_cpy ++;
				ptr_abbt_copy++;
         	}
			ptr_a++;
      	}
   }

	/**
	 * Block Logic: Derives transpose mapping AT = A^T dynamically.
	 * Optimization: Sequentially sets elements leveraging explicit pointers bounded to `j <= i`.
	 */
    for (register int i = 0; i < N; i++) {
		register double *ptr_at = &(AT[i * N]);
        for (register int j = 0; j <= i ; j++) {
            (*ptr_at) = A[j * N + i];
			ptr_at++;
		}
	}

	/**
	 * Block Logic: Evaluates ATA = A^T * A manipulating prior transpose construct.
	 * Optimization: Implements ikj loops to sequentially stride over rows caching inner array offsets.
	 */
    for (register int i = 0; i < N; ++i) {
		register double *ptr_at = &(AT[i * N]);
		register double *ptr_a = &(A[i]);
		register double *ptr_ata = &(ATA[i * N + i]);
        for (register int k = 0; k < N; ++k) {
			register double *ptr_a_cpy = ptr_a + k * N;
			register double *ptr_ata_copy = ptr_ata;
      		for (register int j = i; j < N; ++j) {
            	(*ptr_ata_copy) += (*ptr_at) * (*ptr_a_cpy);
				ptr_a_cpy ++;
				ptr_ata_copy++;
         	}
			ptr_at++;
      	}
   }

	/**
	 * Block Logic: Performs the terminal summation phase merging symmetric boundaries.
	 * Invariant: Generates C = ABBT + ATA avoiding full redundant traversal cycles.
	 */
    for (register int i = 0; i < N; ++i) {
    	for (register int j = i; j < N; ++j) {
			C[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
			C[j * N + i] = ABBT[j * N + i] + ATA[i * N + j];
		}
	}

	free(BBT);
	free(ABBT);
	free(AT);
	free(ATA);

	return C;
}
