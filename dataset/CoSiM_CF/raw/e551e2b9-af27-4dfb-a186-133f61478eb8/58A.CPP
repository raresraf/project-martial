#include <bits/stdc++.h>

using namespace std;

string solve(const string &s) {
  const string hello = "hello";
  int i = 0, j = 0;
  while (i < hello.length() && j < s.length()) {
    if (s[j++] == hello[i])
      i++;
  }
  return (i == hello.length()) ? "YES" : " NO";
}

int main() {
  string s;
  cin >> s;
  cout << solve(s) << endl;
}
