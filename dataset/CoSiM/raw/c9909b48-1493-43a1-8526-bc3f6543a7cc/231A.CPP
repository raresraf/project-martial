#include <stdio.h>

int main() {
  int ara[1000][1000], i, j, x, count = 0, count2 = 0;
  scanf("%d", &x);
  for (i = 0; i < x; i++) {
    for (j = 0; j < 3; j++) {
      scanf("%d", &ara[i][j]);
    }
  }
  for (i = 0; i < x; i++) {
    count = 0;
    for (j = 0; j < 3; j++) {
      if (ara[i][j] == 1) {
        count++;
      }
    }
    if (count >= 2) {
      count2++;
    }
  }
  printf("%d\n", count2);
}
