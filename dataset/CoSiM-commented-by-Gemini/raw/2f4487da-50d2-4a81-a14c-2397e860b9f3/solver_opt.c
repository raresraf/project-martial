
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C = (double *)malloc(N * N * sizeof(double));
    double *D = (double *)malloc(N * N * sizeof(double));
    register int i, j, k;

    
    for (i = 0; i < N; i++) {
    	register double *orig_pb1 = &B[i * N];
        for (j = i; j < N; j++) {
            register double aux = 0;
            register double *pb1 = orig_pb1;
            register double *pb2 = &B[j * N];

            for (k = 0; k < N; k++) {
                aux += *pb1 * *pb2;
                pb1++;
                pb2++;
            }

            C[i * N + j] = aux;
            if (i != j)
                C[j * N + i] = aux;

        }
    }

    
    for (i = 0; i < N; i++) {
    	register double *orig_pa = &A[i * (N + 1)];
        for (j = 0; j < N; j++) {
            register double aux = 0;
            register double *pa = orig_pa;
            register double *pc = &C[j * N + i];

            for (k = i; k < N; k++) {
                aux += *pa * *pc;
                pa++;
                pc++;
            }

            D[i * N + j] = aux;
        }
    }

    
    for (i = 0; i < N; i++) {
    	register double *orig_pa = &A[i];
        for (j = i; j < N; j++) {
            register double aux = 0;
            register double *pa1 = orig_pa;
            register double *pa2 = &A[j];

            for (k = 0; k <= i; k++) {
                aux += *pa1 * *pa2;
                pa1 += N;
                pa2 += N;
            }

            if (i != j) {
                D[j * N + i] += aux;
            }

            D[i * N + j] += aux;
        }
    }

    free(C);
	return D;	
}
