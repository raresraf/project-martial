#include <bits/stdc++.h>

using namespace std;

#define int long long

int t[100];

int32_t main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); ++i)
    if (s[i] >= '0' && s[i] <= '9')
      ++t[s[i] - '0'];
    else
      ++t[10 + s[i] - 'a'];
  int ans = 0;
  for (int i = 0; i < 50; ++i)
    ans += t[i] * t[i];
  cout << ans;
}