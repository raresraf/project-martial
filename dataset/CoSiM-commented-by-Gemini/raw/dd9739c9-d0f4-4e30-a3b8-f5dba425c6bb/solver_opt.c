
#include "utils.h"





void add(double* result, double* A, double* B, int N) {
    for (int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
			*result = *A + *B;
			result++;
			A++;
			B++;
		}
    }
    
};



void prodTriSub(double* result, double* A, double* B, int N) {
	register double* initialA = A; 
	register double* initialB = B; 


    for (register int k = 0; k < N; ++k) {

		register double* A_ptr = initialA; 
		register double* initialRes = result; 

        for (register int i = 0; i < N; ++i) {

			register double* res_ptr = initialRes; 
			register double* B_ptr = initialB; 

            for (register int j = 0; j < N; ++j) {

				if(k >= i) {
					*res_ptr += *A_ptr * *B_ptr; 
				}

				B_ptr++; 
				res_ptr++; 
            }
			A_ptr += N; 
			initialRes += N; 
        }

		initialB += N; 
		initialA++; 
    }

}


void prodTriLow(double* result, double* A, double* B, int N) {
	register double* initialA = A; 
	register double* initialB = B; 


    for (register int k = 0; k < N; ++k) {

		register double* A_ptr = initialA; 
		register double* initialRes = result; 

        for (register int i = 0; i < N; ++i) {

			register double* res_ptr = initialRes; 
			register double* B_ptr = initialB; 

            for (register int j = 0; j < N; ++j) {

				if(k <= i) {
					*res_ptr += *A_ptr * *B_ptr; 
				}

				B_ptr++; 
				res_ptr++; 
            }
			A_ptr += N; 
			initialRes += N; 
        }

		initialB += N; 
		initialA++; 
    }
}



void prod(double* result, double* A, double* B, int N) {
	register double* initialA = A; 
	register double* initialB = B; 


    for (register int k = 0; k < N; ++k) {

		register double* A_ptr = initialA; 
		register double* initialRes = result; 

        for (register int i = 0; i < N; ++i) {

			register double* res_ptr = initialRes; 
			register double* B_ptr = initialB; 

            for (register int j = 0; j < N; ++j) {

				*res_ptr += *A_ptr * *B_ptr; 

				B_ptr++; 
				res_ptr++; 
            }
			A_ptr += N; 
			initialRes += N; 
        }

		initialB += N; 
		initialA++; 
    }
}


double* my_solver(int N, double *A, double* B) {
	double* C = (double*) calloc(N * N, sizeof(double));
	double* At = (double*) calloc(N * N, sizeof(double));
	double* Bt = (double*) calloc(N * N, sizeof(double));

	
	for(register int i = 0; i < N; i++) {
		register double *At_ptr = At + i; 
		register double *Bt_ptr = Bt + i; 

		register double *Ai =  A + i * N; 
		register double *Bi = B + i * N; 

		for(register int j = 0; j < N; j++) {
			*At_ptr = *Ai;
			*Bt_ptr = *Bi;
			At_ptr += N; 
			Bt_ptr += N; 
			Ai++; 
			Bi++; 
		}
	}


	

	double* X = (double*) calloc(N * N, sizeof(double));
	prodTriSub(X, A, B, N);

	
	double* Y = (double*) calloc(N * N, sizeof(double));
	prod(Y, X, Bt, N);

	
	double* Z = (double*) calloc(N * N, sizeof(double));
	prodTriLow(Z, At, A, N);


	
	add(C, Y, Z, N);

	
	free(At);
	free(Bt);
	free(X);
	free(Z);
	free(Y);
	return C;
}
