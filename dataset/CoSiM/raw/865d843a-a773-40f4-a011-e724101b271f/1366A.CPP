
// Problem : A. Shovels and Swords
// Contest : Codeforces - Educational Codeforces Round 89 (Rated for Div. 2)
// URL : https://codeforces.com/contest/1366/problem/A
// Memory Limit : 256.000000 MB
// Time Limit : 1000.000000 milisec

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
  int t;
  cin >> t;
  while (t--) {
    ll n, m;
    cin >> n >> m;
    if (n < m)
      swap(n, m);
    ll total = 0;

    while (n >= 2 && m) {

      ll temp = min((n / 4) + 1, m);
      n -= (temp * 2);
      m -= temp;
      if (temp == 0)
        break;
      total += temp;
      if (n < m)
        swap(n, m);
    }
    cout << total << endl;
  }
  return 0;
}
