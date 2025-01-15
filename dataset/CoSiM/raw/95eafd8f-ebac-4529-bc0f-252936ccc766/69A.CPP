#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  while (cin >> n) {
    int a[n][3];
    int i, j, k, l;
    for (i = 0; i < n; i++) {
      for (j = 0; j < 3; j++) {
        cin >> a[i][j];
      }
    }
    k = l = 0;
    for (j = 0; j < 3; j++) {
      for (i = 0; i < n; i++) {
        l += a[i][j];
      }
      if (l != 0) {
        k = 1;
        break;
      }
    }
    if (k == 0) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
