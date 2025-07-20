#include <cstdio>

using namespace std;

int main() {

  int N;
  fscanf(stdin, "%d", &N);
  int A[N][3];
  int num, suma = 0, sumb = 0;
  int i, j;

  for (i = 0; i < N * 3; ++i) {
    for (j = 0; j < 3; ++j) {
      fscanf(stdin, "%d", &A[i][j]);
    }
  }
  for (i = 0; i < N; ++i) {
    for (j = 0; j < 3; ++j) {
      suma += A[i][j];
    }
  }
  if (suma != 0) {
    fprintf(stdout, "NO\n");
    return 0;
  }
  for (j = 0; j < 3; ++j) {
    for (i = 0; i < N; ++i) {
      sumb += A[i][j];
    }
    if (sumb != 0) {
      fprintf(stdout, "NO\n");
      return 0;
    }
  }
  fprintf(stdout, "YES\n");

  return 0;
}
