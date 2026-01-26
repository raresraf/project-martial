/*
 * @Author: czqsole
 * @Date: 2019-03-12 00:11
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  int l[105] = {0};
  int r[105] = {0};
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> l[i] >> r[i];
  }
  cin >> k;
  int i = 0;
  for (; i < n; i++) {
    if (l[i] <= k && k <= r[i]) {
      break;
    }
  }
  int ans = n - i;
  cout << ans << endl;
}