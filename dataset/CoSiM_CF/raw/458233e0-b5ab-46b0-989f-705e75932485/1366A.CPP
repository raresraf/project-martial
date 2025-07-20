#include <algorithm>
#include <iostream>
using namespace std;

#define ll long long

ll a, b;
ll ans = 0;

bool can(ll n) { return (n * 2 <= a && n <= b) || (n <= a && n * 2 < b); }

void solve(ll a, ll b) {
  ll _max = max(a, b);

  ll _b = 0, _e = _max;

  while (_b <= _e) {
    ll p = _e + (_b - _e) / 2;

    if (can(p)) {
      ans = p;
    }

    if (p > ans) {
      _b = p + 1;
    } else {
      _e = p - 1;
    }
  }
}

int main(void) {
  int t;
  cin >> t;

  while (t--) {
    cin >> a >> b;
    solve(a, b);
    cout << ans << endl;
  }

  return 0;
}