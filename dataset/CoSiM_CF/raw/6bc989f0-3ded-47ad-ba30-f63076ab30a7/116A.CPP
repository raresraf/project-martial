#include <cstdio>

using namespace std;

int main() {
  int N;
  int a, b, curr = 0, max = 0;
  fscanf(stdin, "%d", &N);

  for (int i = 0; i < N; ++i) {
    fscanf(stdin, "%d%d", &a, &b);

    curr = (curr - a) + b;
    if (curr > max) {
      max = curr;
    }
  }
  fprintf(stdout, "%d", max);

  return 0;
}
