#include <bits/stdc++.h>
using namespace std;
int main() {
  int a[99][99], x, y, count, i, j, i1, j1;
  int counti = 0;
  int countj = 0;
  count = 0;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a[i][j];
    }
  }
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      if (a[i][j] == 1) {
        i1 = i;
        j1 = j;
      }
    }
  }
  if (j1 > 3) {
    for (x = j1; x > 3; x--) {
      countj++;
    }
  }
  if (j1 == 3) {
    countj = 0;
  }
  if (j1 < 3) {
    for (x = j1; x < 3; x++) {
      countj++;
    }
  }
  if (i1 > 3) {
    for (y = i1; y > 3; y--)
      counti++;
  }
  if (i1 == 3) {
    counti = 0;
  }
  if (i1 < 3) {
    for (y = i1; y < 3; y++) {
      counti++;
    }
  }
  cout << (counti + countj);
  return 0;
}
