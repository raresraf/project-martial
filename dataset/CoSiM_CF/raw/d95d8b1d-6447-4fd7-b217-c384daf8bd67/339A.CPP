#include <bits/stdc++.h>
using namespace std;
int main(void) {
  // your code goes here
  string s;
  cin >> s;
  int l = s.length();
  int cnt[4] = {0};
  for (int i = 0; i < l; i++) {
    if (s[i] != '+') {
      cnt[s[i] - 48]++;
    }
  }
  string ans = "";
  for (int i = 1; i <= 3; i++) {
    for (int j = 1; j <= cnt[i]; j++) {
      ans += to_string(i) + "+";
    }
  }
  cout << ans.substr(0, ans.length() - 1);
  return 0;
}