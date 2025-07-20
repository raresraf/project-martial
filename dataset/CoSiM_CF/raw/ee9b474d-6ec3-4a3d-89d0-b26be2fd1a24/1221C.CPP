#include <bits/stdc++.h>
using namespace std;

int main() {
  int q;
  long long int c, m, x;
  cin >> q;
  while (q--) {

    cin >> c >> m >> x;

    long long int l = 0, r = min(c, m), ans;
    while (l <= r) {
      long long int mid = (l + r) / 2;
      if ((c + m + x) >= (3 * mid)) {
        l = mid + 1;
        ans = mid;
      } else
        r = mid - 1;
    }
    cout << ans << endl;
  }
}
