#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, n;
  cin >> n >> k;

  int a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  int num = a[k - 1];

  int ans = 0;
  for (int i = 0; i < n; i++)
    if (a[i] >= num && a[i] > 0)
      ans++;
  cout << ans;
  return 0;
}