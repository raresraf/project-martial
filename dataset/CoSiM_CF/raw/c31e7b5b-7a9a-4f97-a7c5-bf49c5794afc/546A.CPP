#include <stdio.h>
using namespace std;

int main() {
  int k, w;
  long n;
  scanf("%d %lu %d", &k, &n, &w);
  long r = (k * (w * (w + 1) / 2) - n);
  r = r > 0 ? r : 0;
  printf("%lu", r);
  return 0;
}
