/// Problem Name: String Task
/// Problem Link: https://codeforces.com/problemset/problem/118/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  string s, ans;
  cin >> s;
  int len = s.size();
  for (int i = 0; i < len; i++) {
    if (s[i] >= 65 && s[i] <= 90)
      s[i] = s[i] + 32;
  }
  for (int i = 0; i < len; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue;
    ans += '.';
    ans += s[i];
  }
  cout << ans << '\n';
  return 0;
}
