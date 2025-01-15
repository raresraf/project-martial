#include <stdio.h>
#include <string.h>
int main() {
  int a[100], b[100], c[100], n, a1 = 0, b1 = 0, c1 = 0, i;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d%d%d", &a[i], &b[i], &c[i]);
    a1 += a[i];
    b1 += b[i];
    c1 += c[i];
  }
  if (a1 == 0 && b1 == 0 & c1 == 0)
    printf("YES");
  else
    printf("NO");
  return 0;
}