#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int res = 0;
  int x = 0;
  int y = 0;
  int matrix[5][5];

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];

      if (matrix[i][j] == 1) {
        x = j;
        y = i;
      }
    }
  }

  res = abs(2 - x) + abs(2 - y);

  cout << res << "\n";

  return 0;
}
