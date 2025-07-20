#include <bits/stdc++.h>
using namespace std;
int main() {
  int a[6][6], i, b, j;
  int n = 5;
  for (i = 1; i <= n; i++) {
    for (j = 1; j <= n; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        b = (abs(3 - i) + abs(3 - j));
      }
    }
  }
  cout << b;
}
