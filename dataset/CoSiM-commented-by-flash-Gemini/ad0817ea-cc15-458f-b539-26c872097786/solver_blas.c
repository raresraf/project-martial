
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {

	double *res1, *res2;
	res1 = malloc(N * N * sizeof(double));
	res2 = malloc(N * N * sizeof(double));

	

	
	
	

	
	
	
	
	dgemm("T","N",N,N,N,1,A,N,A,N,0,res1,N);

	
	dgemm("N","N",N,N,N,1,A,N,B,N,0,res2,N);

	
	dgemm("N","T",N,N,N,1,res2,N,B,N,1,res1,N);

	free(res2);

	return res1;

	printf("BLAS SOLVER\n");
	return NULL;
}
