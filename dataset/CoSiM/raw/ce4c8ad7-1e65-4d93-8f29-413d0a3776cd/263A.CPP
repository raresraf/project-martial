#include <iostream>
using namespace std;
main() {
  int matrix[5][5];
  int x, y;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];
      if (matrix[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }
  x = (x - 2);
  y = (y - 2);
  if (x < 0)
    x *= -1;
  if (y < 0)
    y *= -1;
  int moves = x + y;
  cout << moves << endl;
}
