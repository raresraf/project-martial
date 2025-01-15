#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

//#include <bits/stdc++.h>

using namespace std;

long long max(long long a, long long b) {
  if (a > b)
    return a;
  else
    return b;
}

void solve() {
  int n;
  cin >> n;
  long long ans = 0;
  long mx;
  long x;
  cin >> x;
  mx = x;

  for (int i = 0; i < n - 1; i++) {
    int x;
    cin >> x;

    if (x > 0 == mx > 0)
      mx = max(mx, x);
    else {
      ans += mx;
      mx = x;
    }
  }
  ans += mx;

  cout << ans << "\n";
}

int main() {
  int t;
  cin >> t;
  while (t--)
    solve();

  return 0;
}