#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  vector<int> cnt(t);
  int ans = 0;
  while (t--) {
    int ct = 0;
    for (int i = 0; i < 3; i++) {
      int q;
      cin >> q;
      if (q == 1)
        ct++;
    }
    cnt[t] = ct;
  }
  for (auto i : cnt)
    if (i >= 2)
      ans++;
  cout << ans << endl;
  return 0;
}