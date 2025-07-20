// https://codeforces.com/problemset/problem/266/A

#include <bits/stdc++.h>

using namespace std;

int solve(const string &s) {
  if (s.length() == 0)
    return 0;
  int cnt = 0;
  for (int i = 1; i < s.length(); ++i) {
    if (s[i - 1] == s[i]) {
      cnt++;
    }
  }

  return cnt;
}

int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  cout << solve(s) << endl;
}
