/// Problem Name: Alternating Subsequence
/// Problem Link: https://codeforces.com/problemset/problem/1343/C

/**
 *  winners never quit
 **/

#include <bits/stdc++.h>

using namespace std;

#define pb push_back
#define mp make_pair
typedef long long Long;

const int N = 1e5;

int mx = -((1e9) + 1);

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int tc, ca = 0;
  cin >> tc;
  while (tc--) {
    int n, el;
    cin >> n;
    vector<int> v;
    for (int i = 0; i < n; i++) {
      cin >> el;
      v.pb(el);
    }
    int i = 0;
    Long sum = 0;
    while (i < n) {
      int p_mx = mx;
      while (i < n && v[i] > 0) {
        p_mx = max(p_mx, v[i]);
        i++;
      }
      if (p_mx != mx)
        sum += p_mx;
      int n_mx = mx;
      while (i < n && v[i] < 0) {
        n_mx = max(n_mx, v[i]);
        i++;
      }
      if (n_mx != mx)
        sum += n_mx;
    }
    cout << sum << '\n';
  }
  return 0;
}
