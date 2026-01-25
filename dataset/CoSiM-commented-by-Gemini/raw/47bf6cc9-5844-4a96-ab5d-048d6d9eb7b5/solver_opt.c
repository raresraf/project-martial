
#include "utils.h"



 
double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *C;
	AB = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	if (AB == NULL || C == NULL) {
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }
	
register int i, j, k;
register double *orig_pa, *pa, *pb, sum;


    for (i = 0; i < N; i++) {
		orig_pa = &A[i * N];
		for (j = 0; j < N; j++) {
			pa = orig_pa + i;
			pb = &B[i * N + j];
			sum = 0;
			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb+= N;
			}
			AB[i * N + j] = sum;
		}
	}



	for (i = 0; i < N; i++) {
		orig_pa = &AB[i * N];
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &B[j * N];
			sum = 0;
			for (k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			C[i * N + j] = sum;
		}
	}


	for (i = 0; i < N; i++) {
		orig_pa = &A[i];
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &A[j];
			sum = 0;
			for (k = 0; k < N; k++) {
			    if (*pa == 0 || *pb == 0) {
			        break;
				}
				sum += *pa * *pb;
		        pa += N;
				pb += N;
			}
			C[i * N + j] += sum;
		}
	}

	free(AB);
	return C;
}
