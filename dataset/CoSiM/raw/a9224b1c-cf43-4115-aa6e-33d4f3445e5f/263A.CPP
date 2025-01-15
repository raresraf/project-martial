#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int x = 1, y = 1, i, j, one, moves = 0;

  while (x <= 5) {
    while (y <= 5) {
      scanf("%d", &one);

      if (one == 1) {
        i = x;
        j = y;
      }

      y++;
    }

    y = 1;

    x++;
  }

  if (i < 3) {
    moves += (3 - i);
  } else if (i > 3) {
    moves += (i - 3);
  }

  if (j < 3) {
    moves += (3 - j);
  } else if (j > 3) {
    moves += (j - 3);
  }

  printf("%d\n", moves);

  return 0;
}
