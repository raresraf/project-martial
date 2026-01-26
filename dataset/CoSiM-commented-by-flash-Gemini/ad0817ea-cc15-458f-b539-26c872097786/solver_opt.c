
#include "utils.h"



double* my_solver(int N, double *A, double* B) {

	register int i, j, k;
	double *res1, *res2, *res3;
	register int min;
	register int IN;
	int size = N * N * sizeof(double);
	res1 = malloc(size);
	res2 = malloc(size);
	res3 = malloc(size);

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A[i];	
		IN = i * N;

   		for (j = 0; j < N; j++) {
   			register double *pa = orig_pa;
		    register double *pb = &A[j];		

      		if (j < i) {
      			min = j;
      		} else {
      			min = i;
      		}

      		register double sum1 = 0.0;
			for (k = 0; k <= min; k++) {
				sum1 += *pa * *pb;
			    pa += N;
       			pb += N;
      		}
      		res1[IN + j] = sum1;
   		}
	}

	
	for (i = 0; i < N; i++) {
		IN = i * N;

		register double *orig_pa = &A[IN];	
   		for (j = 0; j < N; j++) {
   			register double *pa = orig_pa;
		    register double *pb = &B[j];		

      		register double sum2 = 0.0;
	      	for (k = 0; k < N; k++) {
	      		sum2 += *pa * *pb;
	       		pa++;
		       	pb += N;
      		}
      		res2[IN + j] = sum2;
   		}
	}

	
	for (i = 0; i < N; i++) {
		IN = i * N;

		register double *orig_pa = &res2[IN];	
   		for (j = 0; j < N; j++) {
   			register double *pa = orig_pa;
		    register double *pb = &B[j * N];		
      		register double sum3 = 0.0;
	      	for (k = 0; k < N; k++) {
	      		sum3 += *pa * *pb;
	       		pa++;
		       	pb++;
      		}
      		res3[IN + j] = sum3;
   		}
	}

	
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res3[i * N + j] += res1[i * N + j];
   		}
	}

	free(res1);
	free(res2);
	

	return res3;

	printf("OPT SOLVER\n");
	return NULL;
}