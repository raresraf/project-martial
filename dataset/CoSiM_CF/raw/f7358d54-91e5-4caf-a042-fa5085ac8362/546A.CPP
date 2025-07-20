#include <stdio.h>
int main() {
  int k, w;
  long long int n, sum = 0;

  scanf("%d %lld %d", &k, &n, &w);

  for (int i = 1; i <= w; i++) {
    sum = sum + (i * k);
  }
  if (sum > n)
    printf("%lld\n", sum - n);
  else
    printf("0\n");
  return 0;
}
