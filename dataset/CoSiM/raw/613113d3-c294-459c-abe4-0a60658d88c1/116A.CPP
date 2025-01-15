#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, a[1000], b[1000], x, t, ans[1000];
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }
  x = a[0] + b[0];
  t = x;
  ans[0] = t;
  for (i = 1; i < n; i++) {
    x = (t - a[i]) + b[i];
    ans[i] = x;
    t = x;
  }
  sort(ans, ans + n);
  cout << ans[n - 1];

  return 0;
}