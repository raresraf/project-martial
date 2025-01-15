#include <bits/stdc++.h>
typedef long long int ll;
#define elif else if
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define rep(i, a, b) for (ll i = a; i < b; i++)
using namespace std;
void solve() {
  ll query;
  cin >> query;
  ll final_ans = 0;
  while (query--) {
    ll ans = 0;
    ll p, q, r;
    cin >> p >> q >> r;
    if (p == 1) {
      ans++;
    }
    if (q == 1) {
      ans++;
    }
    if (r == 1) {
      ans++;
    }
    if (ans >= 2) {
      final_ans++;
    }
  }
  cout << final_ans << endl;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  ll t = 1;
  // cin>>t;
  while (t--) {
    solve();
  }
  return 0;
}