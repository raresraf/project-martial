
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	register int i, j, k;

	printf("OPT SOLVER\n");

	double *at = calloc(N * N, sizeof(double));
	if (at == NULL)
		exit(EXIT_FAILURE);

	double *bt = calloc(N * N, sizeof(double));
	if (bt == NULL)
		exit(EXIT_FAILURE);

	double *res1 = calloc(N * N, sizeof(double));
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	double *res2 = calloc(N * N, sizeof(double));
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	double *res3 = calloc(N * N, sizeof(double));
    if (res3 == NULL)
            exit(EXIT_FAILURE);

	double *res = calloc(N * N, sizeof(double));
	if (res == NULL)
        exit(EXIT_FAILURE);


	
	for (i = 0; i < N; i++) {

		
		register double *p_at = at + i;
		register double *p_bt = bt + i;
		register double *ptA = A + i * N;
		register double *ptB = B + i * N;

		for (j = 0; j < N; j++) {
			*p_at = *ptA;
			*p_bt = *ptB;
			p_at += N;
			p_bt += N;
			ptA++;
			ptB++;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *pt_to_A = &A[i * N + i];
		register double *p_res1 = &res1[i * N];
		for (j = 0; j < N; j++) {
			
			register double sum = 0.0;
			
			register double *pa = pt_to_A;
			register double *pb = &B[i*N + j];
			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			*(p_res1 + j) = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *pt_to_res1 = &res1[i * N];
		for (j = 0; j < N; j++) {
			
			register double sum = 0.0;
			
			register double *pres1 = pt_to_res1;
			register double *pbt = &bt[j];
			for (k = 0; k < N; k++) {
				sum += *pres1 * *pbt;
				pres1++;
				pbt += N;
			}
			res2[i * N + j]= sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		register double *pt_to_at = &at[i * N ];
		for (j = 0; j < N; j++) {
			
			register double sum = 0.0;
			
			register double *pat = pt_to_at;
			register double *pA = &A[j];
			for (k = 0; k <= j; k++) {
				sum += *pat * *pA;
				pat++;
				pA += N;
			}
			res3[i * N + j] = sum ;
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			res[i * N + j] = res2[i * N + j] + res3[i * N + j];
		}
	}

	free(at);
	free(bt);
	free(res1);
	free(res2);
	free(res3);
	return res;
}
