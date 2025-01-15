#include <bits/stdc++.h>
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define ll long long
#define N 100010
using namespace std;

ll t, n, x, a[2], dp[2];

int main() {
  cin >> t;
  while (t--) {
    cin >> n;
    a[0] = a[1] = INT_MIN;
    dp[0] = dp[1] = 0;
    FOR(i, 0, n) {
      cin >> x;
      a[x < 0] = INT_MIN;
      if (a[x > 0] == INT_MIN)
        dp[x > 0] = dp[x < 0] + x;
      else {
        a[x > 0] = max(a[x > 0], x);
        dp[x > 0] = max(dp[x > 0], dp[x < 0] + a[x > 0]);
      }
      a[x > 0] = max(a[x > 0], x);
    }
    cout << dp[x > 0] << '\n';
  }

  return 0;
}