#include <bits/stdc++.h>
using namespace std;
int main() {

  int m, k = 0;
  int a[5][5];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }

  // cout<<a[1][2]<<endl;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if ((a[i][j] == 1) && (i == 4 && j == 4)) {
        k = 4;
      } else if ((a[i][j] == 1) && (i == 3 && j == 3)) {
        k = 2;
      } else if ((a[i][j] == 1) && (i == 1 && j == 1)) {
        k = 2;
      }

      else if ((a[i][j] == 1) && (i == 1 && j == 0)) {
        k = 3;
      } else if ((a[i][j] == 1) && (i == 0 && j == 1)) {
        k = 3;
      } else if ((a[i][j] == 1) && (i == 0 && j == 0)) {
        k = 4;
      } else if ((a[i][j] == 1) && (i == 3 && j == 4)) {
        k = 3;
      } else if ((a[i][j] == 1) && (i == 4 && j == 3)) {
        k = 3;
      }

      else if (a[i][j] == 1) {
        k = (abs(i - j));
      }
    }
  }
  cout << k << endl;
}