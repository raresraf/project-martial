
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C = (double *)calloc(N*N,sizeof(double));
	
	double *AUX = (double *)calloc(N*N, sizeof(double));

	register int bi=0;
    register int bj=0;
    register int bk=0;
    register int i=0;
    register int j=0;
    register int k=0;
	register int blockSize = 40;
	for(bi=0; bi<N; bi+=blockSize)
	{
        for(bj=0; bj<N; bj+=blockSize)
		{
            for(bk=0; bk<N; bk+=blockSize){
                for(i=0; i<blockSize; ++i)
				{
                    for(j=0; j<blockSize; ++j)
					{
						register double res = 0;
                        for(k=0; k<blockSize; ++k){
                            if(bi + i<=bk + k)
							{
								res += A[((bi+i) * N) + (bk+k)] * B[((bk+k) * N) + (bj+j)];
							}
						}
						AUX[((bi+i) * N) + (bj+j)] += res;
					}
				}
			}
		}
	}

	for(bi=0; bi<N; bi+=blockSize)
        for(bj=0; bj<N; bj+=blockSize)
            for(bk=0; bk<N; bk+=blockSize)
                for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++)
					{
						register double res = 0;
                        for(k=0; k<blockSize; k++){
							res += AUX[((bi+i) * N) + (bk+k)] * B[((bj+j) * N) + (bk+k)];
                            if(bk + k<=bj + j){
								res += A[((bk+k) * N) + (bi+i)] * A[((bk+k) * N) + (bj+j)];
							}
						}
						C[((bi+i) * N) + (bj+j)] += res;
					}
				}
	free(AUX);
	return C;
}
