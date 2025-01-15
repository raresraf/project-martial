#include <bits/stdc++.h>

using namespace std;

void solve() {
  int n = 0;
  cin >> n;
  vector<int> a(n);
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  int maximum = INT_MIN;
  int sign = 0;
  long long sum = 0;
  for (int i = 0; i < n; i++) {
    if (sign == 0) {
      if (a[i] > 0) {
        sign = 1;
      } else {
        sign = -1;
      }
      maximum = max(maximum, a[i]);
    } else if (sign == 1) {
      if (a[i] > 0) {
        maximum = max(maximum, a[i]);
      } else {
        sum += maximum;
        maximum = max(INT_MIN, a[i]);
        sign = -1;
      }
    } else {
      if (a[i] > 0) {
        sum += maximum;
        maximum = max(INT_MIN, a[i]);
        sign = 1;
      } else {
        maximum = max(maximum, a[i]);
      }
    }
  }
  sum += maximum;
  cout << sum << '\n';
}

int main() {
  int t = 0;
  cin >> t;
  for (int i = 0; i < t; i++) {
    solve();
  }
  return 0;
}