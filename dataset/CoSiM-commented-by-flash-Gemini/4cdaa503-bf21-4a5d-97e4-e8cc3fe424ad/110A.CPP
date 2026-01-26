/// Problem Name: Nearly Lucky Number
/// Problem Link: https://codeforces.com/problemset/problem/110/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  int d = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '4' || s[i] == '7')
      d++;
  }
  if (d == 4 || d == 7)
    cout << "YES" << '\n';
  else
    cout << "NO" << '\n';
  return 0;
}
