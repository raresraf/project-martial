#include <stdio.h>
int main() {
  int n, a, b, sum = 0, max = 0;
  scanf("%d", &n);

  for (int i = 1; i <= n; i++) {
    scanf("%d %d", &a, &b);
    sum = sum - a;
    sum = sum + b;

    if (sum > max)
      max = sum;
  }
  printf("%d\n", max);
  return 0;
}
