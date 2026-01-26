#include <stdio.h>
#include <stdlib.h>
int main() {
  int m[10][10], row, col, i, j, count = 0;
  row, col;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      scanf("%d", &m[i][j]);
      if (m[i][j] == 1) {
        row = i;
        col = j;
      }
    }
  }
  count += abs(row - 2);
  count += abs(col - 2);
  printf("%d", count);
  return 0;
}