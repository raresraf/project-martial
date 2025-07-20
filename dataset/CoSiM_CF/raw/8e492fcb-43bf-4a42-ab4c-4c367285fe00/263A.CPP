#include <bits/stdc++.h>
using namespace std;

int main() {
  int mat[5][5];
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
  cout << abs(row - 2) + abs(col - 2);
}
