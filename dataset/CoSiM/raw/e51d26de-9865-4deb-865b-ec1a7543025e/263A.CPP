#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace std;

int main() {
  int a[100][100], count = 0, ck = 0;
  int row, column;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }
  int temp;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (a[i][j] == 1) {
        row = i;
        column = j;
      }
    }
  }
  cout << fabs(row - 2) + fabs(column - 2);
}
