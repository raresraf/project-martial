#include <bits/stdc++.h>
using namespace std;
int main() {
  int dp[5][5];
  int row;
  int col;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> dp[i][j];
      if (dp[i][j] == 1) {
        row = i + 1;
        col = j + 1;
      }
    }
  }
  cout << abs(row - 3) + abs(col - 3);

  return 0;
}