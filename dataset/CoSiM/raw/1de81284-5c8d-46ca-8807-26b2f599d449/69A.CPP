/*
Hey there stalker....
















Like what you see?
*/
#include <algorithm>
#include <bits/stdc++.h>
#include <set>
#include <string>
#include <vector>
using namespace std;
#define IOS                                                                    \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define pb push_back
#define pii pair<ll, ll>
#define vi vector<int>
#define all(a) (a).begin(), (a).end()
#define F first
#define S second
#define sz(x) (ll) x.size()
#define rep(i, a, b) for (ll i = a; i < b; i++)
#define repr(i, a, b) for (ll i = a; i > b; i--)
#define lbnd lower_bound
#define ubnd upper_bound
#define bs binary_search
#define mp make_pair
#define ll long long int
#define lld long double
#define endl "\n"
#define MOD 1000000007
int main() {
  IOS

      int TESTS;
  cin >> TESTS;
  ll arr[3] = {0, 0, 0}, count = 0;
  ll a = 0, b = 0, c = 0;
  while (TESTS--) {

    vi v(3);
    rep(i, 0, 3) { cin >> v[i]; }

    a += arr[0] + v[0];
    b += arr[1] + v[1];
    c += arr[2] + v[2];

    if (a == 0 && b == 0 && c == 0) {
      count++;
    }
  }
  // cout << a << b <<c ;
  if (count > 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
