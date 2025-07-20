#include <iostream>

using namespace std;

int main() {
  int m[5][5], i, j, x = 0, y = 0, sum = 0;

  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> m[i][j];

      if (m[i][j] == 1) {
        y = i;
        x = j;
      }
    }
  }

  if (y < 2) {
    sum += 2 - y;
  } else if (y > 2) {
    sum += y - 2;
  } else if (y == 2) {
    sum = sum + 0;
  }
  if (x < 2) {
    sum += 2 - x;
  } else if (x > 2) {
    sum += x - 2;
  } else if (x == 2) {
    sum = sum + 0;
  }

  cout << sum;
}
