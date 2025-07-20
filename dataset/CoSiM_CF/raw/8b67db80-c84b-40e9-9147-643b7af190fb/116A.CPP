#include <stdio.h>
#include <string.h>
int main() {
  int n, i, c;
  scanf("%d", &n);
  int p[n][2];
  for (i = 0; i < n; i++)
    scanf("%d %d", &p[i][0], &p[i][1]);
  int r[n];
  r[0] = p[0][1];
  for (i = 1; i < n - 1; i++) {
    r[i] = r[i - 1] - p[i][0] + p[i][1];
  }
  r[n - 1] = p[n - 1][0];
  c = r[0];
  for (i = 1; i < n; i++) {
    if (r[i] > c)
      c = r[i];
  }
  printf("%d", c);
  return 0;
}
