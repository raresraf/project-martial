#include <bits/stdc++.h>
using namespace std;

int main() {

  int mat[5][5];
  int x, y;
  int n = 5;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      cin >> mat[i][j];
      if (mat[i][j] == 1)
        x = i, y = j;
    }
  cout << abs(2 - x) + abs(2 - y);

  return 0;
}