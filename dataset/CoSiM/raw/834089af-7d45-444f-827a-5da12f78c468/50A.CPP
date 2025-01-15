#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int M, N, area, d;
  scanf("%d %d", &M, &N);
  area = M * N;
  d = area / 2;

  printf("%d\n", d);
}