
#include "utils.h"



double* transpose(int N, double *a){
	double* transpose = (double*)malloc((N * N) * sizeof(double));
	register int i, j;
  		
    for (i = 0; i < N; i++){
		register double *pa = &a[i * N];
    	register double *tb = &transpose[i];

  		for (j = 0; j < N; j++){
  			*tb = *pa;
  			tb += N;
  			pa++;
  		}
	}

  	return transpose;
}

double* add(int N, double *A, double *B){
	register int i, j;

	for(i = 0 ; i < N ; i++){
		register double *pa = &A[i * N];
		register double *pb = &B[i * N];
		for(j = 0; j < N ; j++){
			*pa += *pb;
			pa++;
			pb++;
		}
	}
	return A;
}


double* multiply(int N, double *a) {
	double* c = (double*)calloc(sizeof(double), (N * N));
	register int i, j, k;

	for (i = 0 ; i < N ; i++){
		register double *orig_pa = &a[i * N];
		for (j = 0 ; j < N ; j++){
			register double *pa = orig_pa;
			register double *pb = &a[j * N];
			register double suma = 0;
	    	for (k = 0 ; k < N ; k++){
	    		suma += *pa * *pb;
	    		pa++;
	    		pb++;
	      	}
	      	c[i * N + j] = suma;
   		}
	}

	return c;
}

int min(int a, int b){
	return (a < b) ? a : b;
}

double* multiply_two_triunghiular(int N, double *a, double* b) {
	double* c = (double*)malloc((N * N) * sizeof(double));

	register int i, j, k;

	for (i = 0 ; i < N ; i++){
		register double *orig_pa = &a[i * N];
		for (j = 0 ; j < N ; j++){
			register double *pa = orig_pa;
			register double *pb = &b[j];
			register double suma = 0;
	    	for (k = 0 ; k < min(i + 1, j + 1) ; k++){
	    		suma += *pa * *pb;
	    		pa++;
	    		pb += N;
	      	}
	      	c[i * N + j] = suma;
   		}
	}

	return c;
}

double* multiply_upper_triunghiular(int N, double *a, double* b) {
	double* c = (double*)malloc((N * N) * sizeof(double));

	register int i, j, k;

	for (i = 0 ; i < N ; i++){
		register double *orig_pa = &a[i * N + i];
		for (j = 0 ; j < N ; j++){
			register double *pa = orig_pa;
    		register double *pb = &b[j + N * i];
			register double suma = 0;
	    	for (k = i ; k < N ; k++){
	    		suma += *pa * *pb;
      			pa++;
      			pb += N;
	      	}
	      	c[i * N + j] = suma;
   		}
	}

	return c;
}



double* my_solver(int N, double *A, double* B) {
	double* a_transpose = transpose(N, A);

	double* b_bt = multiply(N, B);

	double* a_b_bt = multiply_upper_triunghiular(N, A, b_bt);

	double* at_a = multiply_two_triunghiular(N, a_transpose, A);

	double* c = add(N, a_b_bt, at_a);

	free(a_transpose);
	free(b_bt);
	free(at_a);

	return c;
}