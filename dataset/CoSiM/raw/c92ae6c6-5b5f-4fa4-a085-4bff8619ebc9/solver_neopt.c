


#include "utils.h"
#include <math.h>




int min(int a,int b){

if(a>b)
	return b;
else
	return a;

}



void mul1 (double *A,double *B,double *C,int N){

int i,j,k;

for(i=0;i<N;i++)
	for(j=0;j<N;j++)
		for(k=i;k<N;k++)  
			C[i*N+j]+=A[i*N+k]*B[k*N+j];

}



void mul2 (double *A,double *B,double *C,int N){

int i,j,k;

for(i=0;i<N;i++)  
	for(j=0;j<N;j++)
		for(k=0;k<N;k++)
			C[i*N+j]+=A[i*N+k]*B[j*N+k];

}



void mul3 (double *A,double *C,int N){

int i,j,k;

for(i=0;i<N;i++)
	for(j=0;j<N;j++)
		for(k=0;k<min(i,j)+1;k++) 
			C[i*N+j]+=A[k*N+j]*A[k*N+i];

}



void add(double *A,double *B,int N){

int i;

for(i=0;i<N*N;i++)
		A[i]+=B[i];

}


double* my_solver(int N, double *A, double* B) {

printf("NEOPT SOLVER\n");


double *C1=(double *)calloc(N*N,sizeof(double));
double *C2=(double *)calloc(N*N,sizeof(double));
double *C3=(double *)calloc(N*N,sizeof(double));


mul1(A,B,C1,N);

mul2(C1,B,C2,N);

mul3(A,C3,N);

add(C2,C3,N);


free(C1);
free(C3);

return C2;
}

