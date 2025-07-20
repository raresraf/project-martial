#include "stdio.h"

int main() {
  int k, n, w;
  scanf("%d%d%d", &k, &n, &w);

  int sum = k * (w * (w + 1) / 2);
  int borrow = sum - n;

  printf("%d", borrow > 0 ? borrow : 0);

  return 0;
}