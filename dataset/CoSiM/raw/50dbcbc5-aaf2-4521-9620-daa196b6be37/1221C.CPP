
#include <bits/stdc++.h>
typedef long long ll;
using namespace std;
int main() {
  cin.tie(nullptr);
  ios::sync_with_stdio(false);
  ll n;
  cin >> n;
  while (n--) {
    ll c, m, x;
    cin >> c >> m >> x;
    if (!c || !m)
      cout << "0\n";
    else {
      ll mn = min(c, m);
      if (c + m + x >= 3 * mn)
        cout << mn << "\n";
      else {
        ll st = 1, ed = 3 * mn, ans = 0;
        while (st <= ed) {
          ll md = (st + ed) / 2;
          if (3 * md > m + c + x) {
            ed = md - 1;
          } else {
            if (ans <= md) {
              ans = md;
              st = md + 1;
            } else
              st = ed + 1;
          }
        }
        cout << ans << "\n";
      }
    }
  }
}