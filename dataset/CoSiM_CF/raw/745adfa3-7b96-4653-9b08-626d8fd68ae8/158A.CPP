#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, i, x, cnt = 0;
  cin >> n >> k;
  vector<int> a(n);
  for (i = 0; i < n; i++)
    cin >> a[i];
  for (i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] and a[i] > 0)
      cnt++;
  }
  cout << cnt;
  return 0;
}
