
#include "utils.h"



double* transpose(int N, double *a){
	double* transpose = (double*)malloc((N * N) * sizeof(double));

	for (int i = 0; i < N; i++)
  		for (int j = 0; j < N; j++)
    		transpose[j*N +i] = a[i*N + j];
  		

  	return transpose;
}

double* multiply(int N, double *a, double* b) {
	double* c = (double*)calloc(sizeof(double), (N * N));

	for (int i = 0 ; i < N ; i++){
		for (int j = 0 ; j < N ; j++){
	    	c[i*N + j] = 0.0;
	    	for (int k = 0 ; k < N ; k++){
				c[i*N + j] += a[i*N + k] * b[k*N + j];
	      	}
   		}
	}

	return c;
}

int min(int a, int b){
	return (a < b) ? a : b;
}

double* multiply_triunghiular(int N, double *a, double* b) {
	double* c = (double*)malloc((N * N) * sizeof(double));

	for (int i = 0 ; i < N ; i++){
		for (int j = 0 ; j < N ; j++){
	    	c[i*N + j] = 0.0;
	    	for (int k = 0 ; k < min(i + 1, j + 1) ; k++){
				c[i*N + j] += a[i*N + k] * b[k*N + j];
	      	}
   		}
	}

	return c;
}
double* add(int N, double *A, double *B){
	for(int i = 0 ; i < N ; i++){
		for(int j = 0; j < N ; j++){
			A[i*N + j] += B[i*N + j];
		}
	}
	return A;
}

double* my_solver(int N, double *A, double* B) {
	double* b_transpose = transpose(N, B);
	double* a_transpose = transpose(N, A);

	double* a_b = multiply(N, A, B);
	double* a_b_b = multiply(N, a_b, b_transpose);

	double* a_a = multiply_triunghiular(N, a_transpose, A);

	double* c = add(N, a_b_b, a_a);

	free(a_transpose);
	free(b_transpose);
	free(a_b);
	free(a_a);

	return c;
}
