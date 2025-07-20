// Problem Link : https://codeforces.com/problemset/problem/231/A

#include <bits/stdc++.h>
#define ll long long

using namespace std;

void solve() {
  ll n;
  cin >> n;
  ll a, b, c, ans = 0;
  while (n--) {
    cin >> a >> b >> c;
    if (a + b + c > 1) {
      ans++;
    }
  }
  cout << ans << endl;
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  solve();

  return 0;
}
