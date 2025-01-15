/// Problem Name: Football
/// Problem Link: https://codeforces.com/problemset/problem/96/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  int len = s.size();
  int d = 0, mx = 1;
  for (int i = 1; i < len; i++) {
    if (s[i] == s[i - 1])
      mx++;
    else {
      d = max(d, mx);
      mx = 1;
    }
  }
  d = max(d, mx);
  if (d > 6)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
