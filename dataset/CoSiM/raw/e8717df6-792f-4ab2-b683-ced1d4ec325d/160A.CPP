#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;

  vector<int> v(n);
  for (auto &it : v)
    cin >> it;

  sort(v.rbegin(), v.rend());
  int s = 0;

  for (auto &n : v)
    s += n;

  int ans = 0;
  int oursum = 0;
  for (int i = 0; i < n; ++i) {
    if (oursum <= s) {
      oursum += v[i];
      s -= v[i];
      ans += 1;
    }
  }

  cout << ans << endl;

  return 0;
}