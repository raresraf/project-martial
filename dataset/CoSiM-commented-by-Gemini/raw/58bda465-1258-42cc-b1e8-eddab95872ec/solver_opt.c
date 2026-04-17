/**
 * @file solver_opt.c
 * @brief High-level source code module.
 * Ensures cache-friendly data access, potential loop unrolling, and SIMD optimizations for C/C++.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	
	double *AB = calloc(N * N, sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	/**
	 * @brief Pre-condition: Evaluates logical divergence based on current state.
	 * Invariant: Guarantees correct execution flow according to conditional partitioning.
	 */
	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N; i++) {
		double *copy_res = &(AB[i * N]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		*copy_res = 0;
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (int k = 0; k < N; k++) {
			double *line = &(A[i * N + k]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			double *col = &(B[k * N]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			double *res = copy_res;
			
			/**
			 * @brief Pre-condition: Evaluates logical divergence based on current state.
			 * Invariant: Guarantees correct execution flow according to conditional partitioning.
			 */
			if (i <= k) {
				/**
				 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
				 * Invariant: Operations within the block strictly maintain target functional boundaries.
				 */
				for (int j = 0; j < N; j += 4) {
					
					*res += *line * *col;
					col++;
					res++;
					*res += *line * *col;
					col++;
					res++;
					*res += *line * *col;
					col++;
					res++;
					*res += *line * *col;
					col++;
					res++;
				}
			}
		}
	}

	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N; i++) {
		double *copy_line = &(AB[i * N]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (int j = 0; j < N; j++) {
			double *line = copy_line;
			double *col = &(B[j * N]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			register double sum = 0.0;
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int k = 0; k < N; k += 4) {
				
				sum += *line * *col;
				line++;
				col++;
				sum += *line * *col;
				line++;
				col++;
				sum += *line * *col;
				line++;
				col++;
				sum += *line * *col;
				line++;
				col++;
			}
			ABBt[i * N + j] = sum;
		}
	}

	
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int k = 0; k < N; k++) {
		double *copy_col = &(A[k * N]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
		/**
		 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
		 * Invariant: Operations within the block strictly maintain target functional boundaries.
		 */
		for (int i = 0; i < N; i++) {
			double *res = &(AtA[i * N]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			double *line = &(A[k * N + i]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
			double *col = copy_col;

			/**
			 * @brief Pre-condition: Evaluates logical divergence based on current state.
			 * Invariant: Guarantees correct execution flow according to conditional partitioning.
			 */
			if (i < k) {
				continue;
			}
			/**
			 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
			 * Invariant: Operations within the block strictly maintain target functional boundaries.
			 */
			for (int j = 0; j < N; j++) {
				
				/**
				 * @brief Pre-condition: Evaluates logical divergence based on current state.
				 * Invariant: Guarantees correct execution flow according to conditional partitioning.
				 */
				if (k <= j) {
					
					*res += *line * *col;
				}
				col++;
				res++;
			}
		}
	}

	
	double *a = &(ABBt[0]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
	double *b = &(AtA[0]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
	double *c = &(C[0]); /* Bitwise/pointer arithmetic for precise data alignment and extraction */
	/**
	 * @brief Pre-condition: Iteration boundaries properly mapped and initialized.
	 * Invariant: Operations within the block strictly maintain target functional boundaries.
	 */
	for (int i = 0; i < N * N; i += 4) {
		
		*c = *a + *b;
		a++;
		b++;
		c++;
		*c = *a + *b;
		a++;
		b++;
		c++;
		*c = *a + *b;
		a++;
		b++;
		c++;
		*c = *a + *b;
		a++;
		b++;
		c++;
	}

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
