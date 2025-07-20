#include <bits/stdc++.h>

using namespace std;

int main() {
  int x, y, n = 0;
  int matrix[5][5];
  for (int a = 0; a < 5; a++) {
    for (int b = 0; b < 5; b++) {
      cin >> matrix[a][b];
    }
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (matrix[j][i] == 1) {
        x = i + 1;
        y = j + 1;
      }
    }
  }
  if (x >= 3) {
    n += x - 3;
  }
  if (x < 3) {
    n += 3 - x;
  }
  if (y >= 3) {
    n += y - 3;
  }
  if (y < 3) {
    n += 3 - y;
  }
  cout << n << endl;
}
