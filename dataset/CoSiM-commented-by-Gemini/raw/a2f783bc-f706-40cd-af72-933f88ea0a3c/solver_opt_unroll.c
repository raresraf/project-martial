/**
 * @file solver_opt_unroll.c
 * @brief Source code module.
 * Intent: Maximize functional utility and performance.
 * Domain-Awareness: Manages execution flow, memory hierarchies, and concurrency. Inferred roles for components based on contextual ambiguity.
 */

#include "utils.h"

void wait_for_input(const char *msg)
{
	printf("%s\n", msg);
	getc(stdin);
}


double* my_solver(int N, double *A, double* B)
{
	double *res = calloc(N * N, sizeof(*res));
	double *tmp1 = calloc(N * N, sizeof(*res));
	double *tmp2 = calloc(N * N, sizeof(*res));
	int blockSize = 40;

	/**
	 * Block Logic: Conditional branch evaluation.
	 * Invariant: Selected branch executed while avoiding invalid states.
	 */
	if (res == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	/**
	 * Block Logic: Conditional branch evaluation.
	 * Invariant: Selected branch executed while avoiding invalid states.
	 */
	if (tmp1 == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	/**
	 * Block Logic: Conditional branch evaluation.
	 * Invariant: Selected branch executed while avoiding invalid states.
	 */
	if (tmp2 == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	
	/**
	 * Block Logic: Iterative loop over elements or bounded range.
	 * Invariant: Loop state and bounds are preserved and advance monotonically.
	 */
	for (register int bi = 0; bi < N; bi += blockSize) {
		
		/**
		 * Block Logic: Iterative loop over elements or bounded range.
		 * Invariant: Loop state and bounds are preserved and advance monotonically.
		 */
		for (register int bj = bi; bj < N; bj += blockSize) {
			/**
			 * Block Logic: Iterative loop over elements or bounded range.
			 * Invariant: Loop state and bounds are preserved and advance monotonically.
			 */
			for (register int bk = 0; bk < N; bk += blockSize) {

				
				/**
				 * Block Logic: Conditional branch evaluation.
				 * Invariant: Selected branch executed while avoiding invalid states.
				 */
				if (bk < bi + blockSize) {
					/**
					 * Block Logic: Iterative loop over elements or bounded range.
					 * Invariant: Loop state and bounds are preserved and advance monotonically.
					 */
					for (register int i = 0; i < blockSize; i++) {
						register double *orig_c = &tmp1[(bi + i) * N + bj]; /* Non-obvious bitwise/pointer op for optimized memory access */
						register double *orig_a = &B[(bi + i) * N + bk]; /* Non-obvious bitwise/pointer op for optimized memory access */
						register double *orig_b = &B[bj * N + bk]; /* Non-obvious bitwise/pointer op for optimized memory access */

						register double *orig_c2 = &tmp2[(bi + i) * N + bj]; /* Non-obvious bitwise/pointer op for optimized memory access */
						register double *orig_a2 = &A[bk * N + bi + i]; /* Non-obvious bitwise/pointer op for optimized memory access */
						register double *orig_b2 = &A[bk * N + bj]; /* Non-obvious bitwise/pointer op for optimized memory access */

						/**
						 * Block Logic: Iterative loop over elements or bounded range.
						 * Invariant: Loop state and bounds are preserved and advance monotonically.
						 */
						for (register int k = 0; k < blockSize; k++) {
							register double *pc1 = orig_c;
							register double *pa1 = orig_a + k;
							register double *pb1 = orig_b + k;

							
							/**
							 * Block Logic: Conditional branch evaluation.
							 * Invariant: Selected branch executed while avoiding invalid states.
							 */
							if (bi + i >= bk + k) {
								register double *pa2 = orig_a2 + k * N;
								register double *pc2 = orig_c2;
								register double *pb2 = orig_b2 + k * N;
								
									
								
								
								
								

								
								

								
								
								
								

								
								
								
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
							} else {
								
								
								
									 
								

								
								
								
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
							}
						}
					}
				} else {
					/**
					 * Block Logic: Iterative loop over elements or bounded range.
					 * Invariant: Loop state and bounds are preserved and advance monotonically.
					 */
					for (register int i = 0; i < blockSize; i++) {
						register double *orig_c = &tmp1[(bi + i) * N + bj]; /* Non-obvious bitwise/pointer op for optimized memory access */
						register double *orig_a = &B[(bi + i) * N + bk]; /* Non-obvious bitwise/pointer op for optimized memory access */
						register double *orig_b = &B[bj * N + bk]; /* Non-obvious bitwise/pointer op for optimized memory access */

						/**
						 * Block Logic: Iterative loop over elements or bounded range.
						 * Invariant: Loop state and bounds are preserved and advance monotonically.
						 */
						for (register int k = 0; k < blockSize; k++) {
							register double *pc1 = orig_c;
							register double *pa1 = orig_a + k;
							register double *pb1 = orig_b + k;

							
							
							
								 
							

							
							
							

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;
						}
					}
				}
			}
		}
	}

	
	/**
	 * Block Logic: Iterative loop over elements or bounded range.
	 * Invariant: Loop state and bounds are preserved and advance monotonically.
	 */
	for (int i = 0; i < N ; i++) {
		/**
		 * Block Logic: Iterative loop over elements or bounded range.
		 * Invariant: Loop state and bounds are preserved and advance monotonically.
		 */
		for (int j = 0; j < N; j++) {
			tmp2[j * N + i] = tmp2[i * N + j];
			tmp1[j * N + i] = tmp1[i * N + j];
		}
	}

	
	/**
	 * Block Logic: Iterative loop over elements or bounded range.
	 * Invariant: Loop state and bounds are preserved and advance monotonically.
	 */
	for (register int bi = 0; bi < N; bi += blockSize) {
		/**
		 * Block Logic: Iterative loop over elements or bounded range.
		 * Invariant: Loop state and bounds are preserved and advance monotonically.
		 */
		for (register int bj = 0; bj < N; bj += blockSize) {
			/**
			 * Block Logic: Iterative loop over elements or bounded range.
			 * Invariant: Loop state and bounds are preserved and advance monotonically.
			 */
			for (register int bk = bi; bk < N; bk += blockSize) {
				/**
				 * Block Logic: Iterative loop over elements or bounded range.
				 * Invariant: Loop state and bounds are preserved and advance monotonically.
				 */
				for (register int i = 0; i < blockSize; i++) {
					register double *orig_c = &res[(bi + i) * N + bj]; /* Non-obvious bitwise/pointer op for optimized memory access */
					register double *orig_a = &A[(bi + i) * N + bk]; /* Non-obvious bitwise/pointer op for optimized memory access */
					register double *orig_b = &tmp1[bk * N + bj]; /* Non-obvious bitwise/pointer op for optimized memory access */
					register double start_k = bi + i - bk < 0 ? 0 : bi + i - bk;

					
					/**
					 * Block Logic: Iterative loop over elements or bounded range.
					 * Invariant: Loop state and bounds are preserved and advance monotonically.
					 */
					for (register int k = start_k; k < blockSize; k++) {
						register double *pc = orig_c;
						register double *pa = orig_a + k;
						register double *pb = orig_b + k * N;

						
						
						
								 
						

						
						
						
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
					}
				}
			}
		}
	}

	
	/**
	 * Block Logic: Iterative loop over elements or bounded range.
	 * Invariant: Loop state and bounds are preserved and advance monotonically.
	 */
	for (int i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative loop over elements or bounded range.
		 * Invariant: Loop state and bounds are preserved and advance monotonically.
		 */
		for (int j = 0; j < N; j++) { 
			res[i * N + j] += tmp2[i * N + j];
		}
	}

	free(tmp1);
	free(tmp2);

	return res;
}
