#include "stdio.h"

int main() {
  int mat[5][5];

  int x = -1;
  int y = -1;

  for (int i = 0; i < 5; i++) {
    scanf("%d%d%d%d%d", &mat[i][0], &mat[i][1], &mat[i][2], &mat[i][3],
          &mat[i][4]);

    if (x > 0)
      continue;

    for (int j = 0; j < 5; j++) {
      if (mat[i][j] == 1) {
        x = i + 1;
        y = j + 1;
        break;
      }
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (mat[i][j] == '1') {
        x = i + 1;
        y = j + 1;
        break;
      }
    }
  }

  int num_x = x - 3;

  if (num_x < 0) {
    num_x *= -1;
  }

  int num_y = y - 3;

  if (num_y < 0) {
    num_y *= -1;
  }

  printf("%d", num_x + num_y);

  return 0;
}