#include <bits/stdc++.h>
#define elif else if
#define pb push_back
#define mp make_pair
#define F first
#define S seccond
typedef long long int ll;
#define rep(i, a, b) for (ll i = a; i < b; i++)
using namespace std;
void solve() {
  ll n;
  cin >> n;
  ll sum = 0;
  vector<ll> v;
  rep(i, 0, n) {
    ll x;
    cin >> x;
    v.push_back(x);
    sum += x;
  }
  ll mid = sum / 2;
  ll pre = mid;
  // cout<<"val of pre is"<<" "<<pre<<endl;
  ll size = v.size();
  sort(v.begin(), v.end());
  reverse(v.begin(), v.end());
  ll counter = 0;
  ll sum1 = 0;
  rep(i, 0, size) {
    sum1 += v[i];
    // cout<<"val of sum1 is"<<" "<<sum1<<endl;
    // cout<<"heloo"<<endl;
    if (sum1 <= pre) {
      // cout<<"hello"<<endl;
      counter++;
      continue;
    } else {
      counter++;
      break;
    }
  }
  cout << counter << endl;
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