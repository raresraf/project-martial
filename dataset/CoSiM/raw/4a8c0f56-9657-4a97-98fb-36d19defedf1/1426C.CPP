/// Problem Name: Increase and Copy
/// Problem Link: https://codeforces.com/problemset/problem/1426/C

#include <bits/stdc++.h>
using namespace std;

#define pb push_back
#define pp pop_back
#define mp make_pair
typedef long long Long;

const int N = 1e5;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  int tc, ca = 0;
  cin >> tc;
  while (tc--) {
    Long n;
    cin >> n;
    if (n <= 2) {
      if (n == 1)
        cout << "0" << '\n';
      else
        cout << "1" << '\n';
      continue;
    }
    Long ans = n;
    Long i = 1;
    Long zero = 0;
    for (; i * i <= n; i++) {
      Long ad = max(zero, i - 2);
      if (n % i && i > 1)
        ad++;
      ans = min(ans, (n / i) + ad);
    }
    ans = min(ans, (n / i) + (i - 1));
    cout << ans << '\n';
  }
  return 0;
}
