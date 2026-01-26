#include <stdio.h>
int main() {
  int i, n, x[105], y[105], z[105], sumx = 0, sumy = 0, sumz = 0;
  scanf("%d", &n);
  for (i = 0; i < n; i++)
    scanf("%d %d %d", &x[i], &y[i], &z[i]);
  for (i = 0; i < n; i++) {
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
  }
  if (sumx == 0 && sumy == 0 && sumz == 0)
    printf("YES");
  else
    printf("NO");
  return 0;
}