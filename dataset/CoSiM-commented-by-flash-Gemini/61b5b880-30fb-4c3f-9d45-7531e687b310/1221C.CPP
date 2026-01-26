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
  int q, c, m, x;
  cin >> q;
  while (q--) {
    cin >> c >> m >> x;
    int mn = min(c, m);
    // int extra=c+m+x-(2*mn);
    cout << min((c + m + x) / 3, mn) << endl;
  }
  return 0;
}
