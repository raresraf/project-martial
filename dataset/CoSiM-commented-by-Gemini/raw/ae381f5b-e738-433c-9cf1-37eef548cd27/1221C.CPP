#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  long long int c, m, x, mn, a = 0;
  cin >> n;
  while (n--) {
    cin >> c >> m >> x;
    mn = min(c, min(m, x));
    if (c == 0 || m == 0 || (c + m + x) < 0)
      cout << 0 << endl;
    else if ((c == 1 && (c + m + x) >= 3) || (m == 1 && (c + m + x) >= 3) ||
             (c && m && (c + m + x) == 3)) {
      cout << 1 << endl;
    } else {
      if (mn == m || mn == c) {
        cout << mn << endl;
      } else {
        // cout<<"Robi"<<endl;
        long long ans = mn;
        c -= mn;
        m -= mn;
        m = max(a, m);
        c = max(a, c);
        ans += min(min(m, c), ((m + c) / 3));
        cout << ans << endl;
      }
    }
  }
  return 0;
}
