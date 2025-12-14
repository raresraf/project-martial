#include <bits/stdc++.h>
using namespace std;

int main() {
  int a[5][5];
  int i, j, m = 0;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        if (i == 0 || i == 4)
          m = m + 2;
        if (i == 1 || i == 3)
          m = m + 1;
        if (j == 0 || j == 4)
          m = m + 2;
        if (j == 1 || j == 3)
          m = m + 1;
      }
    }
  }
  cout << m << endl;
  return 0;
}
