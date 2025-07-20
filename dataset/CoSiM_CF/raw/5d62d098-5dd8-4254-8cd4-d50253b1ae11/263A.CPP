#include <bits/stdc++.h>
using namespace std;

int main() {
  int a[5][5];
  int i, j, c, d, e;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        c = i;
        d = j;
      }
    }
  }
  e = abs(2 - c) + abs(2 - d);
  cout << e;
}