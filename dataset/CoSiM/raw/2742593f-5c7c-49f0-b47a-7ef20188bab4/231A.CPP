#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i, j, c, d = 0;
  cin >> n;
  int q[n][3];
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++)
      cin >> q[i][j];
  }
  for (i = 0; i < n; i++) {
    c = 0;
    for (j = 0; j < 3; j++) {
      if (q[i][j] == 1) {
        c++;
      }
    }
    if (c >= 2)
      d++;
  }
  cout << d << endl;
  return 0;
}
