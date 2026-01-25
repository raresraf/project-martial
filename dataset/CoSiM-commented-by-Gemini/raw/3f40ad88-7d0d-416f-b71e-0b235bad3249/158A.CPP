#include <cstdio>

using namespace std;

int main() {
  int N, k;
  fscanf(stdin, "%d%d", &N, &k);
  int A[N];
  int i;

  for (i = 0; i < N; ++i) {
    fscanf(stdin, "%d", A + i);
  }
  int num = A[k - 1];
  int count = 0;
  for (i = 0; i < N; ++i) {
    if (A[i] >= num && A[i] > 0) {
      ++count;
    }
  }
  fprintf(stdout, "%d", count);

  return 0;
}
