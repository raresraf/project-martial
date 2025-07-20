#include <bits/stdc++.h>

#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define DPT                                                                    \
  for (int i = 0; i <= n; i++) {                                               \
    cout << "dp[" << i << "] = " << dp[i] << endl;                             \
  }
#define CP(s, x) cout << s << " " << x << endl;

using namespace std;
typedef long long ll;
typedef vector<int> vi;

const ll INF = 1e9;
const ll MOD = 1e9 + 7;
const ll MAX = 2e5 + 7;

ll q, c, m, x;

ll Min(ll a, ll b, ll c) { return min(min(a, b), c); }

int main() {
  fastIO;
  cin >> q;
  while (q--) {
    cin >> c >> m >> x;
    ll mn = Min(c, m, x), avg = (c + m + x) / 3;
    if (x <= mn) {
      cout << avg << endl;
    } else {
      cout << mn << endl;
    }
  }

  return 0;
}