
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C1 = calloc(N*N, sizeof(double));
	double *C = calloc(N*N, sizeof(double));


	int bi=0;
    int bj=0;
    int bk=0;
    int i=0;
    int j=0;
    int k=0;
    int blockSize=100; 
 
    for(bi=0; bi<N; bi+=blockSize){
        for(bj=0; bj<N; bj+=blockSize){
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++){
						register double tmp = 0;
                        for(k=0; k<blockSize; k++){
							if(bi+i > bk+k)
								continue;
                        	tmp += A[(bi+i)*N + bk+k] * B[(bk+k)*N + bj+j];
						}
						C1[(bi+i)*N + bj+j] += tmp;
					}
				}
			}
		}
	}


	for(bi=0; bi<N; bi+=blockSize){
        for(bj=0; bj<N; bj+=blockSize){
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++){
						register double tmp = 0;
                        for(k=0; k<blockSize; k++){
                            tmp += C1[(bi+i)*N + bk+k] * B[(bj+j)*N + bk+k];
						}
						C[(bi+i)*N + bj+j] += tmp;
					}
				}
			}
		}
	}

	for(bi=0; bi<N; bi+=blockSize){
        for(bj=0; bj<N; bj+=blockSize){
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++){
						register double tmp = 0;
                        for(k=0; k<blockSize; k++){
							if(bi+i < bk+k)
								continue;
                            tmp += A[(bk+k)*N + bi+i] * A[(bk+k)*N + bj+j];
						}
						C[(bi+i)*N + bj+j] += tmp;
					}
				}
			}
		}
	}

	free(C1);

	return C;
}