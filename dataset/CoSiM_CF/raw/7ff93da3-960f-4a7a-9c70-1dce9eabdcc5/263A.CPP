#include <bits/stdc++.h>
using namespace std;
int main() {
  int a[5][5], r, c, d1, d2;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a[i][j];
    }
  }
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (a[i][j] == 1) {
        r = i;
        c = j;
        break;
      }
    }
  }
  d1 = abs(3 - (r + 1));
  d2 = abs(3 - (c + 1));
  cout << d1 + d2 << endl;
}
