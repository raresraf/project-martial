#include <bits/stdc++.h>
#define ll long long int
using namespace std;
ll ar[311111], br[211111];
vector<ll> v;
map<ll, ll> M;
string s;
ll cnt, Min = 999999999, Max = 0, ans = 0, a, b, c, d, e, k, n, m, x, y, z;
int main() {
  ll test;
  cin >> test;
  while (test--) {
    ans = 0;
    cin >> a >> b >> c;
    cin >> x >> y;
    if (x >= y) {
      if (a > b * 2) {
        ans += b * x;
        a -= b * 2;
        if (a > c * 2) {
          ans += c * y;
        } else {
          while (c > 0) {
            if (c * 2 <= a) {
              ans += c * y;
              break;
            }
            c--;
          }
        }
      } else {
        while (b > 0) {
          if (b * 2 <= a) {
            ans += b * x;
            break;
          }
          b--;
        }
      }
    } else {
      if (a > c * 2) {
        ans += c * y;
        a -= c * 2;
        if (a > b * 2) {
          ans += b * x;
        } else {
          while (b > 0) {
            if (b * 2 <= a) {
              ans += b * x;
              break;
            }
            b--;
          }
        }
      } else {
        while (c > 0) {
          if (c * 2 <= a) {
            ans += c * y;
            break;
          }
          c--;
        }
      }
    }
    cout << ans << endl;
  }

  return 0;
}
