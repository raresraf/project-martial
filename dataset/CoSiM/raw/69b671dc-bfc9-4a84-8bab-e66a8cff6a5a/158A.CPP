#include <stdio.h>
#include <string.h>
int main() {
  int x, y, i, cou = 0;
  scanf("%d %d", &x, &y);
  int c[x];
  for (i = 1; i <= x; i++)
    scanf("%d", &c[i]);
  for (i = 1; i <= x; i++) {
    if (c[i] > 0 && c[i] >= c[y])
      cou = cou + 1;
  }
  printf("%d", cou);

  return 0;
}
