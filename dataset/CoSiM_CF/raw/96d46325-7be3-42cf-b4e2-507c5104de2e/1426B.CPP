/// Problem Name: Symmetric Matrix
/// Problem Link: https://codeforces.com/problemset/problem/1426/B

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
    int n, m;
    cin >> n >> m;
    bool ok = false;
    for (int i = 0; i < n; i++) {
      int a, b, c, d;
      cin >> a >> b >> c >> d;
      if (b == c)
        ok = true;
    }
    string ans = "NO";
    if (ok && (m * m) % 4 == 0)
      ans = "YES";
    cout << ans << '\n';
  }
  return 0;
}
