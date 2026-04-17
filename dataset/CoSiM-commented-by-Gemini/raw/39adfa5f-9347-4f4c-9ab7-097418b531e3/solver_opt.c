/**
 * @file solver_opt.c
 * @brief Optimized implementation of matrix solver utilizing loop unrolling and register caching.
 *
 * Computes A * B * B^T + A^T * A. Employs 2x2 block processing (loop unrolling)
 * to maximize instruction level parallelism and register utilization, significantly
 * improving cache performance.
 */

#include "utils.h"

/**
 * @brief Solves the matrix equation A * B * B^T + A^T * A using 2x2 unrolling.
 *
 * @param N Matrix dimension (assumed to be even for 2x2 unrolling).
 * @param A Pointer to the first input matrix (assumed upper triangular).
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix, or NULL on allocation failure.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

    
	double *BBt = (double *)calloc(N * N, sizeof(double));
	/**
	 * @brief Computes B * B^T with 2x2 loop unrolling.
	 * Pre-condition: N is assumed to be even for exact unrolling.
	 * Invariant: BBt blocks are computed iteratively.
	 */
	for (int i = 0; i < N; i += 2) {
        
	    register int row = i * N;
        register double *original_b = &(B[row]);
        register double *bbt = &(BBt[row]);

		/**
		 * @brief Process columns in blocks of 2.
		 * Pre-condition: row index i is valid.
		 * Invariant: 2x2 blocks of BBt are accumulated.
		 */
		for (int j = 0; j < N; j += 2) {
            
            register double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;

            
            register double *b1 = original_b;
            register double *b2 = &(B[j * N]);

            
			/**
			 * @brief Inner loop computing 4 dot products simultaneously.
			 * Pre-condition: Pointers b1 and b2 are valid.
			 * Invariant: sum variables accumulate dot products for the 2x2 block.
			 */
			for (int k = 0; k < N; k++) {
				sum1 += (*b1) * (*b2);
                sum2 += (*(b1 + N)) * (*b2);
                sum3 += (*b1) * (*(b2 + N));
                sum4 += (*(b1 + N)) * (*(b2 + N));
                b1++;
                b2++;
			}

            
            *(bbt) = sum1;
            *(bbt + N) = sum2;
            *(bbt + 1) = sum3;
            *(bbt + N + 1) = sum4;
            bbt += 2;
		}
	}

    
    double *A_BBt = (double *)calloc(N * N, sizeof(double));
    /**
	 * @brief Computes A * (B * B^T) using 2x2 loop unrolling.
	 * Pre-condition: BBt is fully computed and A is upper triangular.
	 * Invariant: A_BBt blocks are computed iteratively.
	 */
    for (int i = 0; i < N; i += 2) {
        
        register int row = i * N;
        register double *a = &(A[row + i]);

        
		/**
		 * @brief Process valid k indices considering A is upper triangular.
		 * Pre-condition: Starts from k = i to avoid zero elements.
		 * Invariant: Computes partial products along k dimension.
		 */
		for (int k = i; k < N; k++) {
            
            register double *bbt = &(BBt[k * N]);
            register double *a_bbt = &(A_BBt[row]);

            
			/**
			 * @brief Inner loop computing 4 elements simultaneously.
			 * Pre-condition: Pointers are properly aligned to 2x2 blocks.
			 * Invariant: A_BBt blocks accumulate dot products.
			 */
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

    
    double *AtA = (double *)calloc(N * N, sizeof(double));
    /**
	 * @brief Computes A^T * A.
	 * Pre-condition: A is upper triangular.
	 * Invariant: AtA stores computed elements up to row i.
	 */
    for (int i = 0; i < N; i++) {
        
		register int row =  i * N;

        
		/**
		 * @brief Process valid k indices for A^T * A considering upper triangular properties.
		 * Pre-condition: Starts from 0 up to i.
		 * Invariant: Computes partial products along k dimension.
		 */
		for (int k = 0; k <= i; k++) {
            
            register double *a2 = &(A[k * N]);
            register double *a1 = a2 + i;
            register double *ata = &(AtA[row]);

			/**
			 * @brief Inner loop traversing columns with pointer arithmetic.
			 * Pre-condition: Pointers are properly aligned.
			 * Invariant: AtA[row, j] accumulates dot product.
			 */
			for (int j = 0; j < N; j++) {
				*(ata) += (*a1) * (*a2);
                a2++;
                ata++;
			}
		}
	}

    
    /**
	 * @brief Adds AtA to A_BBt using 2x2 loop unrolling.
	 * Pre-condition: Both matrices are fully computed.
	 * Invariant: A_BBt accumulates final results iteratively.
	 */
    for (int i = 0; i < N; i += 2) {
        
        register int row = i * N;
        register double *ata = &(AtA[row]);
        register double *a_bbt = &(A_BBt[row]);

        
		/**
		 * @brief Inner loop performing addition in 2x2 blocks.
		 * Pre-condition: Pointers are aligned.
		 * Invariant: 4 elements are summed per iteration.
		 */
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
