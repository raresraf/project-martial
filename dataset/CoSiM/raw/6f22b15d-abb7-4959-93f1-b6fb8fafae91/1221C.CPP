#include <bits/stdc++.h>
using namespace std;
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  long long int q, i, c, m, x, t, r, xx, u;
  cin >> q;
  for (i = 1; i <= q; i++) {
    u = 0;
    cin >> c >> m >> x;
    t = min(c, m);
    r = (c - t) + (m - t) + x;
    if (r < t) {
      xx = t - r;
      u = xx / 3;
      if (xx % 3 != 0) {
        u++;
      }
    }
    cout << t - u << "\n";
  }
}