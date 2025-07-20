#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  ll k, n, w;

  cin >> k >> n >> w;

  //(k+2*k+3*k+4*k...+)

  ll sum = 0;

  for (ll i = 1; i <= w; i++) {
    sum += i * k;
  }
  ll ans = sum - n;
  if (ans >= 0)
    cout << ans << endl;
  else {
    cout << 0 << endl;
  }
}