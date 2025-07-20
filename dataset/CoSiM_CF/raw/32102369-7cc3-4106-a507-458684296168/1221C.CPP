#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, j, q, c, m, x, ans;
  cin >> q;
  while (q--) {
    cin >> c >> m >> x;
    int l = 0, r = min(c, m);
    while (l <= r) {
      int mid = (l + r) / 2;
      if (c + m + x - 2 * mid >= mid) {
        ans = mid;
        l = mid + 1;
      } else
        r = mid - 1;
    }
    cout << ans << endl;
  }
}
