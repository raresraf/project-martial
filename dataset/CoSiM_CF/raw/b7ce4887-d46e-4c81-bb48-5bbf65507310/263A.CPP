#include <bits/stdc++.h>

using namespace std;

int main() {
  int matrix[5][5];
  int moves;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];

      if (matrix[i][j] == 1) {
        moves = abs(j - 2) + abs(i - 2);
      }
    }
  }

  cout << moves << endl;
}