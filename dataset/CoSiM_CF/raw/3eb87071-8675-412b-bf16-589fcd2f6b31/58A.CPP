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
  string s;
  cin >> s;
  ll len = s.size();
  ll c = 0;
  bool ok = 0;
  ll counter = 0;
  string p = "hello";
  ll len1 = p.size();
  rep(i, 0, len) {
    if (s[i] == p[c]) {
      c++;
      counter++;
      if (counter == len1) {
        ok = 1;
        break;
      }
    }
  }
  //   cout<<"val of counter is"<<" "<<counter<<endl;
  if (ok) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
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