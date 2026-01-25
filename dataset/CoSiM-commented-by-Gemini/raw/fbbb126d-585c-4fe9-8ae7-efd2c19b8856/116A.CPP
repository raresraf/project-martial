#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, x, y, a[1000], entery = 0, leaving = 0;
  int m = -32760;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> x >> y;
    if (i == 0) {
      entery = y - x;
    } else {
      leaving = entery - x;
      entery = leaving + y;
    }

    a[i] = entery;
  }

  for (i = 0; i < n; i++) {
    if (m < a[i]) {
      m = a[i];
    }
  }
  cout << m << endl;
  ;

  return 0;
}
