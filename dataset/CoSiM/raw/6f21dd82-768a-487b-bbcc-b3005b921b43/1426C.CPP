/*
 */
#include <bits/stdc++.h>

using namespace std;

#define ll long long int
#define mp make_pair
#define pb push_back
#define fi first
#define si second
#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
#define F(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cin >> a[i];                                                               \
  }
#define F1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cin >> a[i];                                                               \
  }
#define P(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
#define P1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
#define NF(a, n, m)                                                            \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
#define NF1(a, n, m)                                                           \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
#define PNF(a, n, m)                                                           \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define PNF1(a, n, m)                                                          \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define ceil_div(x, y) (((x) + (y)-1) / (y))
const int nax = 1e7;
const int mod = 1e9 + 7;
ll n;
ll tryit(ll a) {
  if (a >= n) {
    return a;
  }
  ll b = ceil_div(n, a + 1) - 1;
  return a + b;
}

void func() {
  cin >> n;
  ll s = 0, e = 1e10, ans;
  while (s <= e) {
    ll mid = (s + e) / 2;
    ll val = (1 + mid) * (1 + mid);
    if (val >= n) {
      ans = mid >= n ? mid : mid + ceil_div(n, mid + 1) - 1;
      e = mid - 1;
    } else {
      s = mid + 1;
    }
  }
  cout << ans << endl;
}

int main() {
  fastIO
#ifndef ONLINE_JUDGE
      freopen("../inp.txt", "r", stdin);
  freopen("../out.txt", "w", stdout);
#endif
  int t;
  cin >> t;
  while (t--) {
    func();
  }
}