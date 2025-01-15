#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int a[n + 1] = {0};
  int ans = 0;
  for (int i = 1; i <= n; i++) {
    cin >> a[i];
  }
  for (int i = 1; i <= n; i++) {
    if (a[i] >= a[k] && a[i] > 0) {
      ans++;
      // cout<<a[i]<<" ";
    }
  }
  cout << ans;
  return 0;
}