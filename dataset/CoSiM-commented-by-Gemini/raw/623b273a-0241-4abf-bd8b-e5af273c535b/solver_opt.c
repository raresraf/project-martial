/**
 * @file solver_opt.c
 * @brief Cache hit optimization via inline pointers.
 */
#include "utils.h"
#define REG register


/**
 * @brief Optimizes strided accesses into direct pointer increment instructions.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double* result;
	double* AAtranspose;
	double* BBtranspose;
	double* ABBt;
	REG int i, j, k;

	result = calloc(N * N, sizeof(*result));
	AAtranspose = calloc(N * N, sizeof(*AAtranspose));
	BBtranspose = calloc(N * N, sizeof(*BBtranspose));
	ABBt = calloc(N * N, sizeof(*ABBt));

	/**
	 * @pre Blank accumulator array.
	 * @post Generates subproducts of A through sequential memory traversal.
	 */
	for(i = 0; i < N; i++) {
		
		REG double *iA_col_fst = A + i; 

		for(j = 0; j < N; j++) {
			REG double *iA_copy = iA_col_fst;
			
			REG double *iA_col_snd = A + j;
			REG double res = 0;

			for(k = 0; k <= j && k <= i; k++) {
				res += *iA_copy * *iA_col_snd;
				
				/// Jumps N steps explicitly.
				iA_copy += N;
				iA_col_snd += N;
			}
			AAtranspose[i * N + j] = res;
		}
	}

	/**
	 * @pre Empty buffer.
	 * @post Generates BBt product with localized memory caching.
	 */
	for(i = 0; i < N; i++) {
		
		REG double *iB_lin_fst = B + i * N;

		for(j = 0; j < N; j++) {
			REG double *iB_copy = iB_lin_fst;
			
			REG double *iB_lin_snd = B + j * N;
			REG double res = 0;

			for(k = 0; k < N; k++) {
				res += *iB_copy * *iB_lin_snd;
				
				iB_copy++;
				iB_lin_snd++;
			}
			BBtranspose[i * N + j] = res;
		}
	}

	/**
	 * @pre BBtranspose established.
	 * @post Iterates inner pointers producing ABBt intermediate blocks.
	 */
	for(i = 0; i < N; i++) {
		
		REG double *iA_lin_fst = A + i * N;

		for(j = 0; j < N; j++) {
			REG double *iA_copy = iA_lin_fst + i;
			REG double *iB_lin =  BBtranspose + j * N + i;
			REG double res = 0;

			
			for(k = i; k < N; k++) {
				res += *iA_copy * *iB_lin;
				
				iA_copy++;
				iB_lin++;
			}
			ABBt[i * N + j] = res;
		}
	}

	/**
	 * @pre Isolated terms created.
	 * @post Stores total addition directly to output.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			result[i * N + j] = ABBt[i * N + j] + AAtranspose[i * N + j];
		
	}

	free(AAtranspose);
	free(BBtranspose);
	free(ABBt);
	
	return result;	
}
