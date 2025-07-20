#include <iostream>

using namespace std;

int main() {
  int matrix[5][5];
  int pos_x, pos_y;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {

      cin >> matrix[i][j];

      if (matrix[i][j] == 1) {
        pos_x = i;
        pos_y = j;
      }
    }
  }

  int count = 0;

  while (!(pos_x == 2 && pos_y == 2)) {
    if (pos_y < 2) {
      count++;
      pos_y++;
    }

    if (pos_y > 2) {
      count++;
      pos_y--;
    }

    if (pos_x < 2) {
      count++;
      pos_x++;
    }

    if (pos_x > 2) {
      count++;
      pos_x--;
    }
  }

  cout << count << endl;
}