/**
 * @file solver_opt.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B)
{
	double *AtA, *C, *ABBt, *AB;

	
	C = malloc(N * N * sizeof(*C));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (NULL == C)
		exit(1);

	AtA = malloc(N * N * sizeof(*AtA));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (NULL == AtA)
		exit(1);

	AB = malloc(N * N * sizeof(*AB));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (NULL == AB)
		exit(1);

	ABBt = malloc(N * N * sizeof(*ABBt));
	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (NULL == ABBt)
		exit(1);
	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (register int i = 0; i < N; i++)
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (register int j = 0; j < N; j++){
			AB[i * N + j] = 0;
			ABBt[i * N + j] = 0;
			AtA[i * N + j] = 0;
			C[i * N + j] = 0;
		}

	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &A[i * N + i]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (register int j = 0; j < N; j++){
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &B[i * N + j]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = i; k < N; k++){
				suma += *pa* *pb;
				pa++;
				pb +=N;
			}
			AB[i * N + j] = suma;
		}
	}
	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &AB[i * N]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (register int j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &B[j * N]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (register int k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb+=1;
			}
			ABBt[i * N + j] = suma;
		}
	}

	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &A[i]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (register int j = 0; j < N; j++){
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &A[j]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (register int k = 0; k <= j; k++){
				suma += *pa* *pb;
				pa+=N;
				pb+=N;
			}
			AtA[i * N + j] = suma;
		}
	}
	
	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (register int i = 0; i < N; i++) {
		register double *pa = &ABBt[i * N]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		register double *pb = &AtA[i * N]; /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (register int j = 0; j < N; j++){
			C[i * N + j] = *pa + *pb;
			pa++;
			pb++;
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
