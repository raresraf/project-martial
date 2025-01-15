#include <cmath>
#include <iostream>

using std::cin;
using std::cout;
using std::endl;

int main() {
  int matrix[5][5];
  int minMove = 0;
  for (int i = 0; i != 5; ++i) {
    for (int j = 0; j != 5; ++j) {
      cin >> matrix[i][j];
      if (matrix[i][j] == 1) {
        minMove = std::abs(i - 2) + std::abs(j - 2);
      }
    }
  }
  cout << minMove;
  return 0;
}
