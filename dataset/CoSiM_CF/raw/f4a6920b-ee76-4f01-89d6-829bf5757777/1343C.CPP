
// Problem : C. Alternating Subsequence
// Contest : Codeforces - Codeforces Round #636 (Div. 3)
// URL : https://codeforces.com/contest/1343/problem/C
// Memory Limit : 0.000000 MB
// Time Limit : 0.000000 milisec
// Powered by CP Editor (https://github.com/coder3101/cp-editor)

#include <bits/stdc++.h>
#define ll long long int
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define MOD 1000000007
#define vi vector<int>
#define vll vector<ll>
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define eb emplace_back
#define ull unsigned long long
#define bug(x) cout << "  [ " #x << " = " << x << " ]" << endl;
#define KAMEHAMEHA ios_base::sync_with_stdio(0);
#define RASENGAN                                                               \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std;
int main() {
  KAMEHAMEHA
  //    #ifdef _soumik
  //        freopen("input.txt", "r", stdin);
  //    #endif
  int t;
  cin >> t;
  while (t--) {
    ll n;
    cin >> n;
    vll a, filter;
    for (int i = 0, temp; i < n; i++) {
      cin >> temp;
      a.eb(temp);
    }
    ll mx = -1e11, sum = 0;
    bool check = true;
    for (int i = 0; i < n; i++) {
      if (check) {
        if (a[i] < 0) {
          if (mx > -1e10)
            sum += mx;
          mx = -1e11;
          check = false;
        }

      } else {
        if (a[i] > 0) {
          if (mx > -1e10)
            sum += mx;
          mx = -1e11;
          check = true;
        }
      }
      mx = max(a[i], mx);
      // cout<<mx<<endl;
    }
    if (mx > -1e10)
      sum += mx;
    cout << sum << endl;
  }

  return 0;
}
