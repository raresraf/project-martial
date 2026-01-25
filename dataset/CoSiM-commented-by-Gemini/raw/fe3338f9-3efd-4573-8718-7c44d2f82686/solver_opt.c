
#include "utils.h"
#include <string.h>



double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C, *AB, *Bt, *At;
	C =(double *)malloc(N * N * sizeof(double));
	if (C == NULL)
		exit(-1);
	AB =(double *)malloc(N * N * sizeof(double));
	if (AB == NULL)
		exit(-1);
	At =(double *)malloc(N * N * sizeof(double));
	if (At == NULL)
		exit(-1);
	Bt =(double *)malloc(N * N * sizeof(double));
	if (Bt == NULL)
		exit(-1);

	
	for (int i = 0; i < N; ++i) {
		register double *A_tmp = A + i * N;
		register double *B_tmp = B + i * N;

		register double *At_tmp = At + i;
		register double *Bt_tmp = Bt + i;

		for (int j = 0; j < N; ++j, ++A_tmp, ++B_tmp, At_tmp += N, Bt_tmp += N) {
			*At_tmp = *A_tmp;
			*Bt_tmp = *B_tmp;
		}

	}

	
	for (int i = 0; i < N; ++i) {
		
		register double *AB_tmp = AB + i * N;
		for (int j = 0; j < N; ++j, ++AB_tmp) {
			
			register double sum = 0;
			
			register double *A_tmp = A + i * N + i;
			register double *B_tmp = Bt + j * N + i;
			for (int k = i; k < N; ++k, ++A_tmp, ++B_tmp)
				sum += *A_tmp * *B_tmp;
			*AB_tmp = sum;
		}
	}

	
	for (int i = 0; i < N; ++i) {
		
		register double *C_tmp = C + i * N;
		for (int j = 0; j < N; ++j, ++C_tmp) {
			
			register double sum = 0;
			register double *AB_tmp = AB + i * N;
			register double *B_tmp = B + j * N;
			for (int k = 0; k < N; ++k, ++AB_tmp, ++B_tmp)
				sum += *AB_tmp * *B_tmp;
			*C_tmp = sum;
		}
	}


	
	for (int i = 0; i < N; ++i) {
		register double *C_tmp = C + i * N;
		for (int j = 0; j < N; ++j, ++C_tmp) {
			
			register double sum = 0;
			register double *At_tmp = At + i * N;
			register double *A_tmp = At + j * N;
			
			for (int k = 0; k <= i; ++k, ++At_tmp, ++A_tmp)
				sum += *At_tmp * *A_tmp;
			*C_tmp += sum;
		}
	}

	free(AB);
	free(At);
	free(Bt);
	return C;
}
