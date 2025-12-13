
#include "utils.h"


double* my_solver(int N, double *A, double* B) {

	double *C;
	double *AA;
	double *AB;
	double *At;
	double *Bt;

	
	C = calloc(N * N, sizeof(*C));
	AA = calloc(N * N, sizeof(*AA));
	AB = calloc(N * N, sizeof(*AB));
	At = calloc(N * N, sizeof(*At));
	Bt = calloc(N * N, sizeof(*Bt));

	
	for(register int i = 0; i < N; ++i) {
		register double *A_i = A + i;
		register double *B_i = B + i;

		register double *At_i = At + i * N;
		register double *Bt_i = Bt + i * N;

		for (register int j = 0; j < N; ++j) {
			*At_i = *A_i;
			*Bt_i = *B_i;
			At_i ++;
			Bt_i ++;
			A_i += N;
			B_i += N;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		register double *AA_i = AA + i * N; 
		register double *At_pi = At + i * N; 

		for (register int j = 0; j < N; ++j) {
			register double *At_i = At_pi;
			register double *A_i = A + j; 

			register double sum = 0.0;

			for (register int k = 0; k < j + 1; ++k, ++At_i, A_i += N) {
				sum += *At_i * *A_i;	
			}

			*AA_i = sum;
			AA_i ++;
		}
	}


	
	for (register int i = 0; i < N; ++i) {
		register double *AB_i = AB + i * N; 
		register double *A_pi = A + i * N; 

		for (register int j = 0; j < N; ++j) {
			
			register double *A_i = A_pi + i; 
			register double *B_i = B + j + i * N; 

			register double sum = 0.0;

			for (register int k = i; k < N; ++k, ++A_i, B_i += N) {
				sum += *A_i * *B_i;
			}

			*AB_i = sum;
			AB_i ++;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		register double *C_i = C + i * N; 
		register double *AB_pi = AB + i * N; 

		for (register int j = 0; j < N; ++j) {
			register double *AB_i = AB_pi;
			register double *Bt_i = Bt + j; 

			register double sum = 0.0;

			for (register int k = 0; k < N; ++k, ++AB_i, Bt_i += N) {
				sum += *AB_i * *Bt_i;
			}

			*C_i = sum;
			C_i ++;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		register double *C_i = C + i *  N; 
		register double *AA_i = AA + i * N; 

		for (register int j = 0; j < N; ++j, ++AA_i) {
			*C_i += *AA_i;
			C_i ++;
		}
	}


	printf("OPT SOLVER\n");
	
	free(AA);
	free(AB);
	free(At);
	free(Bt);

	return C;		
}
