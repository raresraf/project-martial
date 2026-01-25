

#include "utils.h"



double* multiply(int N, double *A, double *B, int not_sup) {
	double* res = (double*)calloc(N*N, sizeof(double));
	for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++){
                if(k >= i || not_sup){
                    sum = sum + A[i * N + k] * B[k * N + j];
                }
            }
            res[i * N + j] = sum;
        }
    }
	return res;

}

double* add(int N, double *A, double *B) {
	double* res = (double*)calloc(N*N, sizeof(double));
	for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }
	return res;

}

double* transpose(int N, double *A) {
	double* res = (double*)calloc(N*N, sizeof(double));
	for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          int index1 = i*N+j;
          int index2 = j*N+i;
            res[index2] = A[index1];
        }
    }
	return res;

}


double* my_solver(int N, double *A, double* B) {
	double* a_trans = transpose(N, A);
	double* b_trans = transpose(N, B);

	double* a_trans_mult_a = multiply(N, a_trans, A, 1);
	double* a_mult_b = multiply(N, A, B, 0);
	double* a_mult_b_mult_trans = multiply(N, a_mult_b, b_trans, 1);

	double* final = add(N, a_mult_b_mult_trans, a_trans_mult_a);

	free(a_trans);
	free(b_trans);
	free(a_trans_mult_a);
	free(a_mult_b);
	free(a_mult_b_mult_trans);

	

	return final;
	
}





