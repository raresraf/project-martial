#include <stdio.h>
int main() {
  int n, i, a, b, c, count = 0;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d %d %d", &a, &b, &c);
    if (a + b + c == 2 || a + b + c == 3)
      count++;
  }
  printf("%d", count);

  return 0;
}