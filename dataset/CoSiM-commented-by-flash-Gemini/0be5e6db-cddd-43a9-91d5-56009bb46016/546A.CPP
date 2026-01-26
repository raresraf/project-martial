#include <stdio.h>
int main() {
  int k, n, w, i, s = 0, b, c = 1;
  scanf("%d %d %d", &k, &n, &w);
  for (i = 1; i <= w; i++) {
    c = k * i;
    s = s + c;
  }
  b = s - n;
  if (s == n || s < n)
    printf("0");
  else
    printf("%d\n", b);
  return 0;
}