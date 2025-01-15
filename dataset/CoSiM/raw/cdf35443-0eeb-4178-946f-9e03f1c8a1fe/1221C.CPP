#include <bits/stdc++.h>

using namespace std;
typedef long long int ll;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int q;
  cin >> q;
  while (q--) {
    int c, m, x;
    cin >> c >> m >> x;
    if (min({c, m, x}) != x)
      cout << min(c, m) << endl;
    else {
      ll ans = x;
      c -= x;
      m -= x;
      int lo = 0;
      int hi = min(c, m);
      int r = -1;
      while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c - mid + m - mid >= mid) {
          if (mid > r)
            r = mid;
          lo = mid + 1;
        } else
          hi = mid - 1;
      }
      if (r != -1)
        ans += r;
      cout << ans << endl;
    }
  }
}
