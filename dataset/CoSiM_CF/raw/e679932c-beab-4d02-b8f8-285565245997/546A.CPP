#include <stdio.h>

int main() {
  int k, n, w;

  scanf("%d %d %d", &k, &w, &n);
  long long int sum = (n * (2 * k + ((n - 1) * k)) / 2);
  (w >= sum) ? printf("0\n") : printf("%d\n", sum - w);
  // printf("%d\n", sum);
  return 0;
}
