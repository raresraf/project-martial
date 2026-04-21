/**
 * @623b273a-0241-4abf-bd8b-e5af273c535b/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register-level accumulation, and specialized loop bounds for 
 * triangular matrix optimization.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Replaces indirect array indexing with direct pointer increments and 
 * promotes loop-invariant addresses into registers.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *AAt;
	double *BBt;
	double *RESULT;
	register int i, j, k;

	AAt = calloc(N * N, sizeof(*AAt));
	BBt = calloc(N * N, sizeof(*BBt));
	RESULT = calloc(N * N, sizeof(*RESULT));

	/**
	 * Block Logic: Compute BBt = A * B.
	 * Optimization: Uses a local register `suma` to hold intermediate dot products, 
	 * reducing writes to the `BBt` buffer. Pointer arithmetic for matrix traversal.
	 */
	for(i = 0; i < N; i++) {
		double *pa_orig = A + i * N;
		for(j = 0; j < N; j++) {
			register double suma = 0;
			double *pa = pa_orig + i;
			double *pb = B + i * N + j;
			for(k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			BBt[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute RESULT = BBt * B^T.
	 * Optimization: Pointer-based linear sweep for `BBt` and column-major sweep for `B`.
	 */
	for(i = 0; i < N; i++) {
		double *pbbt_orig = BBt + i * N;
		for(j = 0; j < N; j++) {
			register double suma = 0;
			double *pbbt = pbbt_orig;
			double *pb = B + j * N;
			for(k = 0; k < N; k++) {
				suma += *pbbt * *pb;
				pbbt++;
				pb++;
			}
			RESULT[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute AAt = A^T * A.
	 * Optimization: Accumulates the product of A's columns into a local register.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			register double suma = 0;
			double *pa1 = A + i;
			double *pa2 = A + j;
			for(k = 0; k < N; k++) {
				suma += *pa1 * *pa2;
				pa1 += N;
				pa2 += N;
			}
			AAt[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: консолидация результатов (consolidation of results).
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			RESULT[i * N + j] += AAt[i * N + j];
		}
	}

	free(BBt);
	free(AAt);

	return RESULT;
}
