// By archit-dev, contest: Codeforces Round #161 (Div. 2), problem: (A)
// Beautiful Matrix, Accepted, #, Copy
#include <bits/stdc++.h>
using namespace std;

int main() {
  int arr[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      int x;
      cin >> x;
      arr[i][j] = x;
    }
  }
  int c_row, c_col;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (arr[i][j] == 1) {
        c_row = i;
        c_col = j;
      }
    }
  }
  int count = 0;
  while (c_row != 2) {
    if (c_row > 2) {
      for (int i = 0; i < 5; i++) {
        swap(arr[c_row - 1][i], arr[c_row][i]);
      }
      c_row -= 1;
      count++;
    } else {
      for (int i = 0; i < 5; i++) {
        swap(arr[c_row][i], arr[c_row + 1][i]);
      }
      c_row += 1;
      count++;
    }
  }

  while (c_col != 2) {
    if (c_col > 2) {
      for (int i = 0; i < 5; i++) {
        swap(arr[i][c_col - 1], arr[i][c_col]);
      }
      c_col -= 1;
      count++;
    } else {
      for (int i = 0; i < 5; i++) {
        swap(arr[i][c_col], arr[i][c_col + 1]);
      }
      c_col += 1;
      count++;
    }
  }

  cout << count << "\n";
  return 0;
}
