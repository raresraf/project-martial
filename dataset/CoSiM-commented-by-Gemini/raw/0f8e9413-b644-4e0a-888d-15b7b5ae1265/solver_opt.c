
#include "utils.h"




double* my_solver(int N, double *A, double* B) {

	printf("OPT SOLVER\n");
	double *AtA = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));
	double *res = calloc(N * N, sizeof(double));
	
	
	for (register int i = 0; i < N; ++i) {
		
		register double *AtA_ptr 	= AtA + i * N;
		register double *At_ptr 	= A + i;
		
		for (register int j = 0; j < N; ++j) {
			register double *copy_At_ptr = At_ptr;
			
			register double *A_ptr = A + j;
			register double sum = 0;
			
			
			for (register int k = 0; k <= j && k <= i; ++k) { 
				sum += (*copy_At_ptr) * (*A_ptr);
				copy_At_ptr += N;
				A_ptr += N;
			}
			*AtA_ptr = sum;
			AtA_ptr++;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		register double *AB_ptr 	= AB + i * N;
		register double *A_ptr 		= A + i * N + i; 

		for (register int j = 0; j < N; ++j) {
			register double *copy_A_ptr = A_ptr;
			register double *B_ptr = B + i * N + j; 
			register double sum = 0;

			for (register int k = i; k < N; ++k) {
				sum += (*copy_A_ptr) * (*B_ptr);
				copy_A_ptr++;
				B_ptr += N;
			}
			*AB_ptr = sum;
			AB_ptr++;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		register double *AtA_ptr 	= AtA + i * N;
		register double *AB_ptr 	= AB + i * N;
		register double *res_ptr 	= res + i * N;

		for (register int j = 0; j < N; ++j) {
			register double *copy_AB_ptr = AB_ptr;
			register double *Bt_ptr = B + j * N;
			register double sum = 0;

			for (register int k = 0; k < N; ++k) {
				sum += (*copy_AB_ptr) * (*Bt_ptr);
				copy_AB_ptr++;
				Bt_ptr++;
			}
			*res_ptr = sum + *AtA_ptr;
			AtA_ptr++;
			res_ptr++;
		}
	}

	free(AtA);
	free(AB);

	return res;	
}
