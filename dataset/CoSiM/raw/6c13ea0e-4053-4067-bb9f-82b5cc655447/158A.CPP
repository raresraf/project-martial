#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  vector<int> a(n);
  for (auto &it : a)
    cin >> it;
  int ans = 0;
  for (auto it : a)
    if (a[k - 1] <= it and it)
      ans++;
  cout << ans;
  ;
  return 0;
}
