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

  ll arr[12] = {4, 7, 47, 74, 444, 777, 474, 477, 744, 747, 447, 774}, a = 0;

  while (TESTS--) {
    string s;
    cin >> s;
    ll count = 0;
    rep(i, 0, sz(s)) {
      if (s[i] == '4' || s[i] == '7') {
        count++;
      }
    }

    rep(i, 0, 12) {
      if (arr[i] == count) {
        a++;
      }
    }
  }
  if (a > 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
