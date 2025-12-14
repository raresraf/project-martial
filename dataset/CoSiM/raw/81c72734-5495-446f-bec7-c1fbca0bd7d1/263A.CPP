#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, j, ai = 0, aj = 0, a[6][6];
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a[i][j];
      if (a[i][j] != 0) {
        ai = i;
        aj = j;
        break;
      }
    }
  }
  cout << (abs(3 - aj) + abs(3 - ai)) << "\n";
  return 0;
}
