#include <stdio.h>
int main() {
  int n, i, cou = 0;
  scanf("%d", &n);
  int c[n][3];
  for (i = 0; i < n; i++)
    scanf("%d %d %d", &c[i][0], &c[i][1], &c[i][2]);
  for (i = 0; i < n; i++) {
    if (c[i][0] == 1 && c[i][1] == 1 || c[i][0] == 1 && c[i][2] == 1 ||
        c[i][1] == 1 && c[i][2] == 1)
      cou++;
  }
  printf("%d", cou);
  return 0;
}
