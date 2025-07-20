#include <stdio.h>
int main() {
  long long int k, n, w, i, sum = 0;
  scanf("%lld %lld %lld", &k, &n, &w);
  for (i = 1; i <= w; i++) {
    sum += (i * k);
  }
  if (sum - n > 0)
    printf("%lld", sum - n);
  else
    printf("0");
  return 0;
}