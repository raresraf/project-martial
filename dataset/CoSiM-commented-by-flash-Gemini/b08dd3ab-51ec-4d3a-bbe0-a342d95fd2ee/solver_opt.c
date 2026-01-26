
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *AtA = malloc(N * N * sizeof(double));
	double *At = malloc(N * N * sizeof(double));
	double *Bt = malloc(N * N * sizeof(double));
    double *AB = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    int i, j, k;

    double *pat, *pbt, *pa, *pb;
	for (i = 0; i < N; ++i) {
	    pa = A + i * N;
	    pb = B + i * N;
	    pat = At + i;
	    pbt = Bt + i;
		for (j = 0; j < N; ++j) {
            *pat = *pa;
            *pbt = *pb;
            pa++;
            pb++;
            pat += N;
            pbt += N;
        }
    }
    
    double *orig_pa, *orig_pat, *orig_pab;
    double *pc, *pab;
    for(i = 0; i < N; ++i){
	  orig_pat = At + i * N;
	  pc = C + i * N;
	  for(j = 0; j < N; ++j){
	    pat = orig_pat ;
	    pa = At + j * N;
	    register double suma = 0;
	    int kMax = (i < j)? i : j;
	    for(k = 0; k <= kMax; ++k){
	      suma += *pat * *pa;
	      pat++;
	      pa++;
	      
	    }
	    *pc = suma;
	    pc++;
	  }
	}
    
    for(i = 0; i < N; ++i){
	  orig_pa = A + i * N;
	  pab = AB + i * N;
	  for(j = 0; j < N; ++j){
	    pa = orig_pa + i;
	    pb = Bt + j * N + i;
	    register double suma = 0.0;
	    for(k = i; k < N; ++k){
	      suma += *pa * *pb;
	      pa++;
	      pb++;
	      
	    }
	    *pab = suma;
	    pab++;
	  }
	}
	
    for(i = 0; i < N; ++i){
	  orig_pab = AB + i * N;
	  pc = C + i * N;
	  for(j = 0; j < N; ++j){
	    pab = orig_pab;
	    pbt = B + j * N;
	    register double suma = 0.0;
	    for(k = 0; k < N; ++k){
	      suma += *pab * *pbt;
	      pab++;
	      pbt++;
	      
	    }
	    *pc += suma;
	    pc++;
	  }
	}

    free(AtA);
    free(AB);
    free(At);
    free(Bt);
	return C;	
}
