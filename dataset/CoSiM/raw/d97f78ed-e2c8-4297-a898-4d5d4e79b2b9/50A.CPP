#include <cstdio>

using namespace std;

int main() {
  int N, M;

  fscanf(stdin, "%d%d", &N, &M);
  int A[N][M];

  int i, j, count = 0;

  for (i = 0; i < N; ++i) {
    for (j = 0; j < M; ++j) {
      A[i][j] = 0;
    }
  }

  for (i = 0; i < N; ++i) {
    for (j = 0; j < M; ++j) {
      if (A[i][j] == 0 && j + 1 < M && A[i][j + 1] == 0) {
        ++count;
        A[i][j] = A[i][j + 1] = 1;
      } else if (A[i][j] == 0 && j + 1 == M && i + 1 < N && A[i + 1][j] == 0) {
        ++count;
        A[i][j] = A[i + 1][j] = 1;
      }
    }
  }
  fprintf(stdout, "%d", count);
  return 0;
}
