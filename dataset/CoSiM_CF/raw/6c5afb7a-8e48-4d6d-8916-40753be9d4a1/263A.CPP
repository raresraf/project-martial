#include <iostream>
using namespace std;

int main() {
  int a[5][5], i, j, c = 0, x = 0, y = 0;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      scanf("%d", &a[i][j]);
      if (a[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }
  if (x < 2 && x != 2) {
    while (x < 2 && x != 2) {
      x++;
      c++;
    }
  } else if (x > 2 && x != 2) {
    while (x > 2 && x != 2) {
      x--;
      c++;
    }
  }
  if (y < 2 && y != 2) {
    while (y < 2 && y != 2) {
      y++;
      c++;
    }
  } else if (y > 2 && y != 2) {
    while (y > 2 && y != 2) {
      y--;
      c++;
    }
  }
  cout << c;
}
