// CF https://codeforces.com/problemset/problem/231/A

#include <bits/stdc++.h>

using namespace std;

int32_t main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  int t;
  cin >> t;
  int ans = 0;
  while (t--) {
    int sum = 0, f = 3;
    while (f--) {
      int x;
      cin >> x;
      sum += x;
    }
    if (sum > 1) {
      ans++;
    }
  }
  cout << ans << "\n";
  return 0;
}