#include <bits/stdc++.h>
using namespace std;

int main() {
  int mat[6][6], row, col;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> mat[i][j];
      if (mat[i][j]) {
        row = i;
        col = j;
      }
    }
  }
  int t = fabs(row - 2) + fabs(col - 2);
  cout << t << endl;

  return 0;
}
