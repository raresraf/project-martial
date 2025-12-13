
#include "utils.h"

double* my_solver(int n, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C = (double *)calloc(n * n, sizeof(double));
	double *D = (double *)calloc(n * n, sizeof(double));

	register int bi=0;
    register int bj=0;
    register int bk=0;
    register int i=0;
    register int j=0;
    register int k=0;

	int blockSize = 25;
	
	
	for(bi = 0; bi < n; bi+=blockSize) {
		for(bj = 0; bj < n; bj+=blockSize) {
			for(bk = 0; bk < n; bk+=blockSize) {
				for(i = 0; i < blockSize; i++) {
					
					register double *orig_pbt = &B[(bi+i) * n + bk];

					for(j = 0; j < blockSize; j++) {

						register double *pbt = orig_pbt;
						register double *pb = &B[(bj + j) * n + bk];

						register double suma = 0.0;

						for(k = 0; k < blockSize; k++) {

							suma += *pbt * *pb;

							pbt++;
							pb++;
						}
						D[(bi+i) * n + bj + j] += suma;
					}
				}
			}
		}
	}

	
	
    for(bi = 0; bi < n; bi+=blockSize) {
        for(bj = 0; bj < n; bj+=blockSize) {
            for(bk = bi; bk < n; bk+=blockSize) {

				register int start = 0;

                for(i = 0; i < blockSize; i++) {
					
					register double *orig_pa;

					if(bk + blockSize - bi - i < 0) {
						start = i;
						orig_pa = &A[(bi + i) * n + bk + i];
					} else {
					    orig_pa = &A[(bi + i) * n + bk];
					}

                    for(j = 0; j < blockSize; j++) {

						register double *pa = orig_pa;
   						register double *pd = &D[(bk  + start) * n + bj + j];

						register double suma = 0.0;
					
                        for( k = start; k < blockSize; k++) {

						   suma += *pa * *pd;

     					   pa++;
                           pd += n;
						}
						C[(bi+i) * n + bj + j] += suma;
					}

				}
			}
		}
	}

	
	
	 for(bi = 0; bi < n; bi+=blockSize) {
		register double end = bi;
        for(bj = bi; bj < n; bj+=blockSize) {
            for(bk = 0; bk < n; bk+=blockSize) {
                for(i = 0; i < blockSize; i++) {
					end++;
					register double *orig_pat = &A[bk * n + bi + i];
					for(j = 0; j < blockSize; j++) {

						if (bj + j < bi + i) {
							continue;
						}
						
						register double *pat = orig_pat;
   						register double *pa = &A[bk * n  + bj + j];
						
						register double suma = 0.0;

                        for(k = 0; k < blockSize && bk + k <= end; k++) {

                            suma +=  *pat * *pa;

							pat += n;
                            pa += n;
						}
						if (bi + i != bj + j) {
							C[(bj+j) * n + bi + i] += suma;
						} 
						C[(bi+i) * n + bj + j] += suma;
					}

				}
			}
		}
	}

	free(D);
	
	return C;	
}
