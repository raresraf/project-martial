/* winners never quit and quitters never win.
      #swap                                 */
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
typedef long double ld;
#define mod 1000000007
#define inf 1000000000000000000
#define rk 256
#define pb push_back
#define mp make_pair
#define vc vector
#define fs first
#define sec second
#define pq priority_queue
#define lb lower_bound
#define ub upper_bound
#define pll pair<ll, ll>
#define pls pair<ll, string>
#define psl pair<string, ll>
#define plc pair<ll, char>
#define pcl pair<char, ll>
#define pss pair<string, string>
#define all(v) v.begin(), v.end()
#define tol(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define tou(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define T                                                                      \
  ll t = 0;                                                                    \
  cin >> t;                                                                    \
  for (ll test = 0; test < t; test++)

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  // isprime();
  // coeff();
  /*freopen("input.txt", "r", stdin);
freopen("output.txt", "w", stdout);*/
  ll n;
  cin >> n;
  vc<pll> v(n);
  for (ll a = 0; a < n; a++)
    cin >> v[a].fs >> v[a].sec;
  sort(all(v));
  ll k;
  cin >> k;
  for (ll a = 0; a < n; a++) {
    if (v[a].sec >= k) {
      cout << n - a << endl;
      break;
    }
  }
  return 0;
}
