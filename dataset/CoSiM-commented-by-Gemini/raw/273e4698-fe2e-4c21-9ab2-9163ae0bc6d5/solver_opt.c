
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
    printf("OPT SOLVER\n");
	double *C, *At, *Bt, *BBt; 
	register int i, j, k;
	
	C    = calloc(N * N, sizeof(*C));
	At   = calloc(N * N, sizeof(*At));
	Bt   = calloc(N * N, sizeof(*Bt));
	BBt  = calloc(N * N, sizeof(*BBt));

	if((C == NULL) || (At == NULL) || (Bt == NULL) || (BBt == NULL)) {
        	return NULL;
    }
    
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			if(i >= j)
				At[i * N + j] = A[j * N + i];
			Bt[i * N + j] = B[j * N + i];
		}
	}

	
	for (i = 0; i < N; i ++) {
        register double *orig_pa = &At[i * N];
        for(j = 0; j < N; j ++) {
            register double *pa = orig_pa;
            register double *pb = &A[j];
            register double suma = 0;

            for(k = 0; k <= j; k++) {
	            suma += (*pa) * (*pb);
                pa ++;
                pb += N;
            }
            C[i * N + j] = suma;
        }
    }

    
    for (i = 0; i < N; i ++) {
        register double *orig_pa = &B[i * N];
        for(j = 0; j < N; j ++) {
            register double *pa = orig_pa;
            register double *pb = &Bt[j];
            register double suma = 0;
            for(k = 0; k < N; k++) {
                suma += (*pa) * (*pb);
                pa ++;
                pb += N;
            }
            BBt[i * N + j] = suma;
        }
    }

    
    
    for (i = 0; i < N; i ++) {
        register double *orig_pa = &A[i * N];
        for(j = 0; j < N; j ++) {
            register double *pa = orig_pa + i;
            register double *pb = &BBt[j] + i * N;
            register double suma = 0;
            for(k = i ; k < N; k++) {
                suma += (*pa) * (*pb);
                pa ++;
                pb += N;
            }
            C[i * N + j] += suma;
        }
    }

	free(At);
	free(Bt);
	free(BBt);
    
	return C;	
}
