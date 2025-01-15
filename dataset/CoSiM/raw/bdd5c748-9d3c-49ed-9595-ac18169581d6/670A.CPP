#include <bits/stdc++.h>

using namespace std;
using ll = long long;
using vll = vector<ll>;
using vvll = vector<vll>;
using vvvll = vector<vvll>;

#define REP(i, n, m) for (ll i = n; i < (ll)m; ++i)
#define IREP(i, n, m) for (ll i = n - 1; i >= m; --i)
#define rep(i, n) REP(i, 0, n)
#define irep(i, n) IREP(i, n, 0)
#define all(v) v.begin(), v.end()
#define vprint(v)                                                              \
  for (auto e : v) {                                                           \
    cout << e << " ";                                                          \
  };                                                                           \
  cout << endl;
#define vvprint(vv)                                                            \
  for (auto v : vv) {                                                          \
    vprint(v)                                                                  \
  };

int main() {
  cin.tie(0);
  ios::sync_with_stdio(false);
  cout << setprecision(20);

  ll t;
  cin >> t;

  rep(i, t) {
    ll n;
    cin >> n;
    vll a(n);
    rep(j, n) cin >> a[j];
    unordered_map<ll, ll> m;
    rep(j, n)++ m[a[j]];
    unordered_set<ll> b, c;
    for (auto p : m) {
      if (p.second >= 2) {
        b.insert(p.first);
        c.insert(p.first);
      } else
        c.insert(p.first);
    }
    ll c1 = 0, c2 = 0;
    while (b.count(c1) > 0)
      ++c1;
    while (c.count(c2) > 0)
      ++c2;
    cout << c1 + c2 << endl;
  }
}
