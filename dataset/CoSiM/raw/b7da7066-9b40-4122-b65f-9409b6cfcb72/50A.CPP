#include <stdio.h>
int main() {
  int m, n, k;
  scanf("%d%d", &m, &n);
  if (m * n >= 2) {
    if (m % 2 == 0 && n % 2 != 0)
      printf("%d", n * (m / 2));
    if (n % 2 == 0 && m % 2 != 0)
      printf("%d", m * (n / 2));
    if (m % 2 != 0 && n % 2 != 0) {
      k = ((n - 1) / 2) * m;
      k = k + ((m - 1) / 2);
      printf("%d", k);
    }
    if (m % 2 == 0 && n % 2 == 0)
      printf("%d", ((m * n) / 2));

  } else
    printf("0");
  return 0;
}