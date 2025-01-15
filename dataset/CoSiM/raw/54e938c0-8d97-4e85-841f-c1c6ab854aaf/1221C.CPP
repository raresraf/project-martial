#include <bits/stdc++.h>
using namespace std;

int n;
int main() {
  int q, c, m, x;
  cin >> q;
  while (q--) {
    cin >> c >> m >> x;
    int res = 0;
    int mi = min(c, m);
    int ma = max(c, m);
    x += ma - mi;
    res += min(x, mi);

    mi -= x;
    if (mi > 0) {
      res += (mi * 2) / 3;
    }

    cout << res << endl;
  }
}

// 4 4         4 0   2
// 2 3         2 1   1
// 3 6         3 3   3
// 3 4         3 1   2
// 2 10        2 8   2

// min(ma/2, mi);
