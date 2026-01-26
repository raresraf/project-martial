#include <stdio.h>
int main() {
  int n, i, k = 0, S = 0, a, b;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d%d", &a, &b);
    S = S - a;
    S = S + b;
    if (k <= S)
      k = S;
  }
  printf("%d", k);
  return 0;
}