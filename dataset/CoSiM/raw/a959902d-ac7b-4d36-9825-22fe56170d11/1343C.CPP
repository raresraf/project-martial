// link:
// [prajwal1721]
// g++ -Wall -Wextra -Wshadow filename.cpp
/*comments:
 */

#include <bits/stdc++.h>
using namespace std;
#define mod 1000000007
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a * b) / gcd(a, b)
#define bits(x) __builtin_popcountll(x)

typedef long long int ll;
void solve() {
  ll t, n;
  cin >> n;
  vector<ll> v;
  for (ll i = 0; i < n; i++) {
    cin >> t;
    v.push_back(t);
  }
  ll s = 0, f = -1, cur = v[0];
  ll i = 0;
  while (i < n) {
    if (f == 1 || v[i] > 0 && i == 0) {
      cur = v[i];
      while (v[i] > 0 && i < n) {
        if (v[i] > cur) {
          cur = v[i];
        }
        i++;
      }
      s += cur;
      f = 0;
    } else {
      cur = v[i];
      while (v[i] < 0 && i < n) {
        if (v[i] > cur) {
          cur = v[i];
        }
        i++;
      }
      s += cur;
      f = 1;
    }
  }
  cout << s << "\n";
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  ll t;
  cin >> t;
  // cout<<t<<"Hi test complete\n";
  while (t--) {
    solve();
  }

  return 0;
}