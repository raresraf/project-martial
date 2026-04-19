/**
 * @raw/ad0817ea-cc15-458f-b539-26c872097786/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A^T * A + (A * B) * B^T.
 * Algorithm: Improves loop ordering taking cache latency constraints into consideration by utilizing explicitly tracked variables.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to auxiliary storage buffers.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {

	register int i, j, k;
	double *res1, *res2, *res3;
	register int min;
	register int IN;
	int size = N * N * sizeof(double);
	
	/**
	 * Functional Utility: Provisions intermediary matrices enforcing clean bounds tracking elements.
	 */
	res1 = malloc(size);
	res2 = malloc(size);
	res3 = malloc(size);

	/**
	 * Block Logic: Solves AtA = A^T * A.
	 * Optimization: Extends local memory optimization logic avoiding sequential recalculation delays defining arrays individually inside execution.
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A[i];	
		IN = i * N;

   		for (j = 0; j < N; j++) {
   			register double *pa = orig_pa;
		    register double *pb = &A[j];		

      		if (j < i) {
      			min = j;
      		} else {
      			min = i;
      		}

      		register double sum1 = 0.0;
			for (k = 0; k <= min; k++) {
				sum1 += *pa * *pb;
			    pa += N;
       			pb += N;
      		}
      		res1[IN + j] = sum1;
   		}
	}

	/**
	 * Block Logic: Generates res2 = A * B mapping index strides locally using predefined array elements.
	 * Optimization: Decreases execution footprint substituting indexing steps with additive pointer bounds.
	 */
	for (i = 0; i < N; i++) {
		IN = i * N;

		register double *orig_pa = &A[IN];	
   		for (j = 0; j < N; j++) {
   			register double *pa = orig_pa;
		    register double *pb = &B[j];		

      		register double sum2 = 0.0;
	      	for (k = 0; k < N; k++) {
	      		sum2 += *pa * *pb;
	       		pa++;
		       	pb += N;
      		}
      		res2[IN + j] = sum2;
   		}
	}

	/**
	 * Block Logic: Evaluates the sub-component producing res3 = res2 * B^T effectively resolving (A * B) * B^T.
	 * Optimization: Sequences cache allocations extracting identical constants externally to mitigate inner processing operations.
	 */
	for (i = 0; i < N; i++) {
		IN = i * N;

		register double *orig_pa = &res2[IN];	
   		for (j = 0; j < N; j++) {
   			register double *pa = orig_pa;
		    register double *pb = &B[j * N];		
      		register double sum3 = 0.0;
	      	for (k = 0; k < N; k++) {
	      		sum3 += *pa * *pb;
	       		pa++;
		       	pb++;
      		}
      		res3[IN + j] = sum3;
   		}
	}

	/**
	 * Block Logic: Solves the equation consolidating previously completed structures rendering final vector inside res3.
	 * Invariant: Executes sequential iterations traversing contiguous cache elements linearly mitigating spatial cache misses.
	 */
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res3[i * N + j] += res1[i * N + j];
   		}
	}

	free(res1);
	free(res2);
	

	return res3;

	printf("OPT SOLVER\n");
	return NULL;
}
