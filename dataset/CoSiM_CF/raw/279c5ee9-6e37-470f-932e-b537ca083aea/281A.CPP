/// Problem Name: Word Capitalization
/// Problem Link: https://codeforces.com/problemset/problem/281/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  if (s[0] >= 97)
    s[0] = s[0] - 32;
  cout << s << '\n';
  return 0;
}
