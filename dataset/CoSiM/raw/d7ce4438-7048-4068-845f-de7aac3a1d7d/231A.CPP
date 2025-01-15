#include <cstdio>

using namespace std;

int main() {
  int N;
  fscanf(stdin, "%d", &N);
  int num, count = 0, fcount = 0;

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < 3; ++j) {
      fscanf(stdin, "%d", &num);
      if (num == 1) {
        ++count;
      }
    }
    if (count >= 2) {
      ++fcount;
    }
    count = 0;
  }

  fprintf(stdout, "%d", fcount);
  return 0;
}
