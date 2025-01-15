#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  unsigned long long int k, w, n, cost;

  while (scanf("%llu%llu%llu", &k, &w, &n) != EOF) {
    cost = (n * (n + 1) * k) / 2;
    if (cost > w)
      printf("%llu\n", cost - w);
    else
      printf("0\n");
  }

  return 0;
}
