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
#define vi vector<ll>
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

      int TESTS = 1;
  // cin >> TESTS;
  while (TESTS--) {
    string s;
    cin >> s;
    string hel = "hello";
    ll j = 0, count = 0;
    rep(i, 0, sz(s)) {
      if (s[i] == hel[j]) {
        j++;
        count++;
      }
    }
    if (count == 5) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
