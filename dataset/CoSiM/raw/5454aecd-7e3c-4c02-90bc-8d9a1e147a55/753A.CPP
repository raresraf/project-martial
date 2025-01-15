//          * AUTHOR : Divyanshu *

#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define f(a, b) for (ll i = a; i < b; i++)
#define mod 1000000007
#define mp make_pair
#define pb push_back
#define vll vector<ll>
#define pll vector<pair<ll, ll>>
#define ld long double
#define fr(a, b) for (ll j = a; j >= b; j--)
#define fi(a, b) for (ll j = a; j < b; j++)
#define fii(a, b) for (ll k = a; k < b; k++)

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "rt", stdin);
// freopen("output.txt", "wt", stdout); *No Need*
#endif
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(0);

  ll n;
  cin >> n;
  if (n == 1) {
    cout << 1 << "\n1";
    return 0;
  } else if (n == 2) {
    cout << "1\n2";
    return 0;
  }
  unordered_map<ll, ll> m;
  ll k = 1;
  vll v;
  while (n) {
    if (!m[n - k] && !m[k] && k != n - k) {
      v.pb(k);
      m[k]++;
      n -= k;
      k++;
    } else {
      v.pb(n);
      n = 0;
    }
  }
  cout << v.size() << "\n";
  f(0, v.size()) cout << v[i] << " ";

#ifndef ONLINE_JUDGE
  cout << "\nTime Elapsed : " << 1.0 * clock() / CLOCKS_PER_SEC << " s\n";
#endif
}
