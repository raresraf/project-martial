#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int mat[6][6];
  int row, col;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> mat[i][j];
      if (mat[i][j] == 1) {
        row = i;
        col = j;
      }
    }
  }
  int res = abs(2 - row) + abs(2 - col);
  cout << res << endl;
}
