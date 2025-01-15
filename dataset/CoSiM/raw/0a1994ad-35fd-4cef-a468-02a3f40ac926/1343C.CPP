#include <bits/stdc++.h>

#define fastread                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define ll long long
#define pb push_back
#define pairs pair<ll, ll>
#define vll vector<ll>
#define all(x) x.begin(), x.end()
#define MAX 200003

using namespace std;

ll MOD = 1e9 + 7;
ll INF = 1000000000000000005;

ll add(ll a, ll b) { return ((a % MOD) + (b % MOD)) % MOD; }
ll mul(ll a, ll b) { return ((a % MOD) * (b % MOD)) % MOD; }
ll sub(ll a, ll b) { return ((a % MOD) - (b % MOD) + MOD) % MOD; }

ll t, n, i, a[MAX], flag, sum, cur;

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  fastread;

  cin >> t;
  while (t--) {
    cin >> n;
    for (i = 1; i <= n; i++) {
      cin >> a[i];
    }

    if (a[1] < 0)
      flag = 0;
    else
      flag = 1;
    cur = a[1];
    i = 2;
    sum = 0;

    while (i <= n) {
      if ((a[i] < 0 && flag == 0) || (a[i] > 0 && flag == 1)) {
        cur = max(cur, a[i]);
      } else {
        sum += cur;
        cur = a[i];
        flag = 1 - flag;
      }
      i++;
    }
    sum += cur;
    cout << sum << "\n";
  }
  return 0;
}