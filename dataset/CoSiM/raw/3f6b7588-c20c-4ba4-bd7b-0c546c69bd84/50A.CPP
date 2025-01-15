#include <stdio.h>
int main() {
  int M, N, maxnum;
  scanf("%d%d", &M, &N);
  maxnum = M * N;
  maxnum = maxnum / 2;
  printf("%d\n", maxnum);
  return 0;
}