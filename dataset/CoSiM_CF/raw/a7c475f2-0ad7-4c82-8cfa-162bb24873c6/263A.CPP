#include <iostream>
using namespace std;

int main() {
  const int pos = 2;
  int matrix[5][5], x, y, result = 0;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];
      if (matrix[i][j] == 1) {
        x = i;
        y = j;
      }
    }

  if (x > 2)
    while (x != 2) {
      ++result;
      --x;
    }
  else if (x < 2)
    while (x != 2) {
      ++result;
      ++x;
    }
  if (y > 2)
    while (y != 2) {
      ++result;
      --y;
    }
  else if (y < 2)
    while (y != 2) {
      ++result;
      ++y;
    }

  cout << result << endl;

  return 0;
}