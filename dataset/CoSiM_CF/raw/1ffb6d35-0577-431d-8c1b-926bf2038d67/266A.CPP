/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/266/A
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, c(0);
  string s;
  cin >> n >> s;
  for (int i = 0; i < n; i++) {
    if (s[i] == s[i + 1])
      c++;
  }
  cout << c;
  return 0;
}
