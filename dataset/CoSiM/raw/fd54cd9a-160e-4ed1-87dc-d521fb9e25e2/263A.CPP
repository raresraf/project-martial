#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int mat[5][5];
  int row = 0, col = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> mat[i][j];
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (mat[i][j] == 1) {

        row = i;
        col = j;
      }
    }
  }
  int x, y;

  if (row == 2) {
    x = 0;
  } else if (row < 2) {
    x = 2 - row;
  }

  else if (row > 2) {
    x = row - 2;
  }

  if (col == 2) {
    y = 0;
  } else if (col < 2) {
    y = 2 - col;
  }

  else if (col > 2) {
    y = col - 2;
  }

  cout << x + y;
}
