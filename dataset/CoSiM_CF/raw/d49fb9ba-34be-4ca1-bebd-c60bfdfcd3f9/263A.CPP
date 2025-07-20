#include <bits/stdc++.h>
using namespace std;

int main() {
  int matrix[6][6], i, j, moves;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      scanf("%d", &matrix[i][j]);
      if (matrix[i][j] == 1) {
        moves = abs(i - 3) + abs(j - 3);
      }
    }
  }
  cout << moves;
}
