#include <bits/stdc++.h>
using namespace std;
int main() {
  int arr[6][6], row, col, i, j, counts = 0;
  for (row = 1; row <= 5; row++) {
    for (col = 1; col <= 5; col++) {
      cin >> arr[row][col];
    }
  }
  for (row = 1; row <= 5; row++) {
    for (col = 1; col <= 5; col++) {
      if (arr[row][col] == 1) {
        i = row;
        j = col;
        break;
      }
    }
  }
  row = i;
  col = j;
  while (row < 3) {
    row++;
    counts++;
  }
  while (col < 3) {
    col++;
    counts++;
  }
  while (row > 3 && row <= 5) {
    row--;
    counts++;
  }
  while (col > 3 && col <= 5) {
    col--;
    counts++;
  }
  cout << counts << endl;

  return 0;
}
