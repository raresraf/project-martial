#include <stdio.h>

int abs(int number) {
  if (number < 0)
    return number * -1;
  return number;
}

int main() {
  int matrix[6][6];
  int xPos = 0;
  int yPos = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      scanf("%d", &matrix[i][j]);
      if (matrix[i][j] == 1) {
        xPos = j;
        yPos = i;
      }
    }
  }

  int howMany = abs(2 - xPos) + abs(2 - yPos);
  printf("%d\n", howMany);

  return 0;
}
