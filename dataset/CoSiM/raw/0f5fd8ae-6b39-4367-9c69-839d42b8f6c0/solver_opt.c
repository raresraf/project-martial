
#include "utils.h"




double *transpose_matrix (int N, double *A) {
	int i, j;
	double *A_t;

	A_t = calloc(N * N, sizeof(double));
	if (!A_t) {
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < N; ++i) {
		
		register double *A_t_ptr = A_t + i;
		
		register double *A_ptr = A + i * N;

		for (j = 0; j < N; ++j) {
			*A_t_ptr = *A_ptr;
			A_t_ptr += N;
			A_ptr++;
		}
	}

	return A_t;
}


double* AB(int N, double *A, double *B) {
	int i, j, k;
	double *AB;
	
	AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(EXIT_FAILURE);
	}

	

	for (i = 0; i + 1 < N; i+=2) {
		
		register double *ptr_AB_i = AB + i * N;
		
		register double *ptr_AB_i1 = AB + (i + 1) * N;
		
	 	register double *ptr_A = A + i * (N + 1);
		for (k = i; k < N; k++) {
			
			register double *ptr_AB_0 = ptr_AB_i;
			
			register double *ptr_AB_1 = ptr_AB_i1;
			
			register double *ptr_B = B + k * N;
			
			register double cons_value_i = *ptr_A;
			
			register double cons_value_i1 = *(ptr_A + N);
			for (j = 0; j + 1 < N; j += 2) {
				
				*ptr_AB_0 += cons_value_i * *ptr_B;
				
				*(ptr_AB_0 + 1) += cons_value_i * *(ptr_B + 1);
				
				*ptr_AB_1 += cons_value_i1 * *ptr_B;
				
				*(ptr_AB_1 + 1) += cons_value_i1 * *(ptr_B + 1);
				ptr_B += 2;
				ptr_AB_0 += 2;
				ptr_AB_1 += 2;
			}
			ptr_A++;
		}
	}

	return AB;
}


double* ABB_t(int N, double *AB, double *B) {
	int i, j, k;
	double *ABB_t;
	
	ABB_t = calloc(N * N, sizeof(double));
	if (!ABB_t) {
		exit(EXIT_FAILURE);
	}

	double *B_t = transpose_matrix(N, B);

	

	for (i = 0; i < N; i += 2) {
		
		register double *ptr_AB = AB + i * N;
		for (k = 0; k < N; k++) {
			
			register double *ptr_ABBt_0 = ABB_t + i * N;
			
			register double *ptr_ABBt_1 = ABB_t + (i + 1) * N;
			
			register double *ptr_B = B_t + k * N;
			
			register double cons_value_i = *ptr_AB;
			
			register double cons_value_i1 = *(ptr_AB + N);
			for (j = 0; j + 1 < N; j += 2) {
				
				*ptr_ABBt_0 += cons_value_i * *ptr_B;
				
				*(ptr_ABBt_0 + 1) += cons_value_i * *(ptr_B + 1);
				
				*ptr_ABBt_1 += cons_value_i1 * *ptr_B;
				
				*(ptr_ABBt_1 + 1) += cons_value_i1 * *(ptr_B + 1);
				ptr_B += 2;
				ptr_ABBt_0 += 2;
				ptr_ABBt_1 += 2;
			}
			ptr_AB++;
		}
	}

	free(B_t);
	return ABB_t;
}


double* A_tA(int N, double *A_t, double *A) {
	int i, j, k;
	double *A_tA;
	
	A_tA = calloc(N * N, sizeof(double));
	if (!A_tA) {
		exit(EXIT_FAILURE);
	}

	

	double *A_tt = transpose_matrix(N, A);

	for (i = 0; i < N; i += 2) {
		
		register double *ptr_AtA = A_tt + i * N;
		for (k = 0; k < N; k++) {
			
			register double *ptr_C_0 = A_tA + i * N;
			
			register double *ptr_C_1 = A_tA + (i + 1) * N;
			
			register double *ptr_A = A + k * N;
			
			register double cons_value_i = *ptr_AtA;
			
			register double cons_value_i1 = *(ptr_AtA + N);
			for (j = 0; j + 1 < N; j += 2) {
				
				*ptr_C_0 += cons_value_i * *ptr_A;
				
				*(ptr_C_0 + 1) += cons_value_i * *(ptr_A + 1);
				
				*ptr_C_1 += cons_value_i1 * *ptr_A;
				
				*(ptr_C_1 + 1) += cons_value_i1 * *(ptr_A + 1);
				ptr_A += 2;
				ptr_C_0 += 2;
				ptr_C_1 += 2;
			}
			ptr_AtA++;
		}
	}

	free(A_tt);
	return A_tA;
}


double* sum(int N, double *A, double *B) {
	int i;
	double *C;
	
	C = calloc(N * N, sizeof(double));
	if (!C) {
		exit(EXIT_FAILURE);
	}

	
	register double *ptr_A = A;
	register double *ptr_B = B;
	register double *ptr_C = C;
	for (i = 0; i < N * N; i++) {
		*ptr_C = *ptr_A + *ptr_B;
		ptr_A++;
		ptr_B++;
		ptr_C++;
	}

	return C;
}


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register double *ptr_AB = AB(N, A, B);
	register double *ptr_ABB_t = ABB_t(N, ptr_AB, B);
	register double *ptr_A_tA = A_tA(N, A, A);
	double *ptr_sum = sum(N, ptr_ABB_t, ptr_A_tA);

	free(ptr_AB);
	free(ptr_ABB_t);
	free(ptr_A_tA);
	
	return ptr_sum;
}
