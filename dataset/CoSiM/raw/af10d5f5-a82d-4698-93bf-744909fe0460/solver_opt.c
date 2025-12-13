
#include "utils.h"


double *allocate_matrix(int N) {
	double *res = calloc(N * N, sizeof(double));
	return res;
}

double* my_solver(int N, double *A, double* B) {
	double *C = allocate_matrix(N);
	double *AxB = allocate_matrix(N);
	double *AxBxBt = allocate_matrix(N);
	double *AtxA = allocate_matrix(N);
    register int i, j, k;

    for (i = 0; i < N; i++) {
        register double *orig_pa = A + N * i + i;   
        for (j = 0; j < N; j++) { 
            register double *pa = orig_pa;
            register double *pb = B + N * i + j;
            register double sum = 0;
            for (k = i; k < N; k++) {
                sum += *pa * *pb;
                pa++;
                pb += N;
            }
            *(AxB + i * N + j) = sum;
        }
    } 

    for (i = 0; i < N; i++) {
        register double *orig_pa = AxB + N * i;           
        for (j = 0; j < N; j++) {
            register double *pa = orig_pa;
            register double *pb = B + j * N;
            register double sum = 0;
            for (k = 0; k < N; k += 1) {
                sum += *pa * *pb;
                pa++;
                pb++;
            }
            *(AxBxBt + i * N + j) = sum;
        }
    } 

    for (i = 0; i < N; i++) {
        register double *orig_pa = A + i;         
        for (j = 0; j < N; j++) {
            register double *pa = orig_pa;
            register double *pb = A + j;            
            register double sum = 0;
            for (k = 0; k <= i; k++) {
                sum += *pa * *pb;
                pa += N;
                pb += N;                                   
            }
            *(AtxA + i * N + j) = sum;
        }
    } 

	for (i = 0; i < N; i++) {
        register double *orig_pa = AtxA + i * N;   
        register double *pa = orig_pa;
        register double *pb = AxBxBt + i * N; 
		for (j = 0; j < N; j++) {
			*(C + i * N + j) = *pa + *pb;
            pa++;
            pb++;
		}
	}

	free(AxB);
	free(AxBxBt);
	free(AtxA);
	return C;
}
