#include <bits/stdc++.h>
using namespace std;

int matrix[5][5];

int main() {

  int i, j, x, y;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      scanf("%d", &matrix[i][j]);
      if (matrix[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }
  printf("%d\n", (abs(x - 2) + abs(y - 2)));

  return 0;
}
