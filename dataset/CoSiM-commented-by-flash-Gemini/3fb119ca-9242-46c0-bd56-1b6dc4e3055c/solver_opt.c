
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation using pre-transposition and pointer arithmetic.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs multiple low-level C tuning strategies:
 * 1. Explicit Pre-transposition: Converts column-major access patterns into 
 *    sequential row-major increments, maximizing spatial locality and cache hit rates.
 * 2. Pointer Arithmetic: Uses direct pointer increments (pA++, pB+=N) instead 
 *    of repeated index calculations.
 * 3. Register Caching: Frequently used pointers and loop variables are marked 
 *    for register allocation to minimize load/store latencies.
 * 4. Triangular Awareness: Reduces dot product range for operations involving A.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

#define min(x, y) (((x) < (y)) ? (x) : (y))

/**
 * my_solver - Optimized matrix solver with cache-aware traversal.
 */
double* my_solver(int N, double *A, double *B) {
	register double * AB, *ABBt, * AtA, * result, *Bt, *At;
	register int i, j, k;

	// Logic: Space allocation for results, workspace, and transposed copies.
	AB = malloc(N * N * sizeof(double));
	ABBt = malloc(N * N * sizeof(double));
	AtA = malloc(N * N * sizeof(double));
	result = malloc(N * N * sizeof(double));
	Bt = malloc(N * N * sizeof(double));
	At = malloc(N * N * sizeof(double));

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits upper triangularity by starting the inner loop 
	 * at k=i. Uses register-backed pointers for dot product accumulation.
	 */
	for(i = 0; i < N; i++) {
		register double *orig_pA = A + i * N;;
		for(j = 0; j < N; j++) {
			register double *pA = orig_pA + i;
   			register double *pB = B + i * N + j;
   			register double current_result = 0;
   			for(k = i; k < N; k++) {
   				current_result += *pA * *pB;
   				pA += 1;
   				pB += N;
   			}
			AB[i * N + j] = current_result;
		}
	}

	/**
	 * Block Logic: Explicit Transposition phase.
	 * Optimization: Pre-calculates A^T and B^T to enable linear memory 
	 * traversal in subsequent matrix multiplication steps.
	 */
	for (i = 0; i < N; i++) {
		register double *pA = A + i * N;  
		register double *pAt = At + i;
		register double *pB = B + i * N;   
		register double *pBt= Bt + i;   
		for (j = 0; j < N; j++) {
			*pAt = *pA;
			*pBt = *pB;
			pAt += N;
			pBt += N;
			pA += 1;
			pB += 1;
		}
	}

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Optimization: Uses the pre-computed Bt to allow both operands to be 
	 * traversed via linear pointer increments in the innermost loop.
	 */
	for(i = 0; i < N; i++) {
		register double *orig_pAB = AB + i * N;;
		for(j = 0; j < N; j++) {
			register double *pAB = orig_pAB;
   			register double *pBt = Bt + j;
   			register double current_result = 0;
   			for(k = 0; k < N; k++) {
   				current_result += *pAB * *pBt;
   				pAB += 1;
   				pBt += N;
   			}
			ABBt[i * N + j] = current_result;
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Leverages At and the upper-triangular limit min(i, j).
	 */
	for(i = 0; i < N; i++) {
		register double *orig_pAt = At + i * N;;
		for(j = 0; j < N; j++) {
			register double *pAt = orig_pAt;
   			register double *pA = A + j;
   			register double current_result = 0;
   			for(k = 0; k <= min(i, j); k++) {
   				current_result += *pAt * *pA;
   				pAt += 1;
   				pA += N;
   			}
			AtA[i * N + j] = current_result;
		}
	}

	/**
	 * Block Logic: Linear result aggregation.
	 * Logic: Final element-wise summation using pure sequential pointer increments.
	 */
	for (i = 0; i < N; ++i) {
		register double *pABBt = ABBt + N * i;
		register double *pAtA = AtA + N * i;
		register double *p_result = result + N * i;
		for (j = 0; j < N; ++j) {
			*p_result = *pABBt + *pAtA;
			p_result += 1;
			pABBt += 1;
			pAtA += 1;
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	free(At);
	free(Bt);
	return result;
}
