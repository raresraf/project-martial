#include <bits/stdc++.h>
using namespace std;

int main() {
  int a[5][5], row, col;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        row = i;
        col = j;
      }
    }
  }
  int ans = abs(row - 2) + abs(col - 2);
  cout << ans;
  return 0;
}
