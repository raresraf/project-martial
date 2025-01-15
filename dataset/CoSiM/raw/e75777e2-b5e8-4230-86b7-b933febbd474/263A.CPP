#include <stdio.h>

int main() {
  int a[5][5], i, j, b, c, d;
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++) {
      scanf("%d", &a[i][j]);
      if (a[i][j] == 1) {
        b = i + 1;
        c = j + 1;
      }
    }
  if (b <= 3)
    d = 3 - b;
  else
    d = b - 3;
  if (c <= 3)
    d = d + 3 - c;
  else
    d = d + c - 3;
  printf("%d", d);
  return 0;
}