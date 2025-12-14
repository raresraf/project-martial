#include <bits/stdc++.h>
using namespace std;

typedef long long int ll;

int main() {
  // #ifndef ONLINE_JUDGE
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  // #endif

  auto sgn = [&](int x) {
    if (x > 0) {
      return 1;
    } else {
      return -1;
    }
  };

  int t;
  cin >> t;
  while (t--) {
    int n;
    ll total = 0;
    cin >> n;
    std::vector<int> v(n);
    for (auto &it : v)
      cin >> it;
    for (int i = 0; i < n; ++i) {
      int cur = v[i];
      int j = i;
      while (j < n && sgn(v[i]) == sgn(v[j])) {
        cur = max(cur, v[j]);
        ++j;
      }
      total += cur;
      i = j - 1;
    }
    cout << total << "\n";
  }

  return 0;
}